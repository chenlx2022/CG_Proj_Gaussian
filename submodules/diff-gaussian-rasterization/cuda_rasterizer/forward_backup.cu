/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
// =====================================================
// Refactored version: Separates basis computation from weighting
// This makes the SH evaluation process more explicit and easier
// to understand for educational purposes
// =====================================================
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
    // 根据相机的观察方向（campos -> pos），将 3D 高斯的球谐系数（SH）展开为 RGB 颜色。
    // 支持 0 到 3 阶展开，阶数越高，颜色随视角的变化越精细。
	
	// =====================================================
	// Step 1: Compute view direction
	// =====================================================
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh_ptr = ((glm::vec3*)shs) + idx * max_coeffs;

	// =====================================================
	// Step 2: Evaluate spherical harmonic basis functions
	// Store basis values in an array for clarity
	// =====================================================
	float basis[16];
	
	// Degree 0 (constant term)
	basis[0] = SH_C0;

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;

		// Degree 1 (linear terms)
		basis[1] = -SH_C1 * y;
		basis[2] = SH_C1 * z;
		basis[3] = -SH_C1 * x;

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			// Degree 2 (quadratic terms)
			basis[4] = SH_C2[0] * xy;
			basis[5] = SH_C2[1] * yz;
			basis[6] = SH_C2[2] * (2.0f * zz - xx - yy);
			basis[7] = SH_C2[3] * xz;
			basis[8] = SH_C2[4] * (xx - yy);

			if (deg > 2)
			{
				// Degree 3 (cubic terms)
				basis[9] = SH_C3[0] * y * (3.0f * xx - yy);
				basis[10] = SH_C3[1] * xy * z;
				basis[11] = SH_C3[2] * y * (4.0f * zz - xx - yy);
				basis[12] = SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
				basis[13] = SH_C3[4] * x * (4.0f * zz - xx - yy);
				basis[14] = SH_C3[5] * z * (xx - yy);
				basis[15] = SH_C3[6] * x * (xx - 3.0f * yy);
			}
		}
	}

	// =====================================================
	// Step 3: Compute weighted sum of SH coefficients
	// result = sum(sh_i * basis_i)
	// =====================================================
	glm::vec3 result = sh_ptr[0] * basis[0];
	int num_coeffs_active = (deg + 1) * (deg + 1);

	for (int i = 1; i < num_coeffs_active; ++i)
	{
		result += sh_ptr[i] * basis[i];
	}

	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
// =====================================================
// Optimized version: Manual expansion of matrix operations
// Computes Sigma' = J * W * Sigma * W^T * J^T
// where J is the Jacobian of perspective projection
// and W is the world-to-camera rotation matrix
// =====================================================
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
    // EWA Splatting 算法的核心实现：
    // 1. 将 3D 中心投影到相机坐标系。
    // 2. 计算透视投影的 Jacobian 矩阵 J。
    // 3. 获取相机变换矩阵 W。
    // 4. 计算变换矩阵 T = W * J。
    // 5. 应用公式 Sigma' = T^T * Sigma * T 得到屏幕空间的 2D 协方差矩阵。
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// =====================================================
	// Step 1: Compute Jacobian J (sparse 2x3 matrix)
	// J represents the derivative of projection with respect
	// to 3D camera coordinates
	// =====================================================
	float J00 = focal_x / t.z;
	float J02 = -(focal_x * t.x) / (t.z * t.z);
	float J11 = focal_y / t.z;
	float J12 = -(focal_y * t.y) / (t.z * t.z);
	// J is sparse: [J00  0  J02]
	//              [ 0  J11 J12]

	// =====================================================
	// Step 2: Extract rotation matrix W from viewmatrix
	// Viewmatrix is column-major, so we extract rows
	// =====================================================
	float3 W_row0 = { viewmatrix[0], viewmatrix[4], viewmatrix[8] };
	float3 W_row1 = { viewmatrix[1], viewmatrix[5], viewmatrix[9] };
	float3 W_row2 = { viewmatrix[2], viewmatrix[6], viewmatrix[10] };

	// =====================================================
	// Step 3: Compute T = J * W (only top 2 rows matter)
	// T_row0 = J00 * W_row0 + J02 * W_row2
	// T_row1 = J11 * W_row1 + J12 * W_row2
	// =====================================================
	float3 T0 = {
		J00 * W_row0.x + J02 * W_row2.x,
		J00 * W_row0.y + J02 * W_row2.y,
		J00 * W_row0.z + J02 * W_row2.z
	};

	float3 T1 = {
		J11 * W_row1.x + J12 * W_row2.x,
		J11 * W_row1.y + J12 * W_row2.y,
		J11 * W_row1.z + J12 * W_row2.z
	};

	// =====================================================
	// Step 4: Compute 2D covariance Sigma' = T * Sigma * T^T
	// Sigma (3D covariance) is symmetric, stored as upper triangle
	// =====================================================
	float V00 = cov3D[0], V01 = cov3D[1], V02 = cov3D[2];
	float V11 = cov3D[3], V12 = cov3D[4], V22 = cov3D[5];

	// Compute temp_0 = T0 * V (vector-matrix product)
	float3 V_T0 = {
		T0.x * V00 + T0.y * V01 + T0.z * V02,
		T0.x * V01 + T0.y * V11 + T0.z * V12,
		T0.x * V02 + T0.y * V12 + T0.z * V22
	};

	// Compute 2D covariance elements
	float cov00 = V_T0.x * T0.x + V_T0.y * T0.y + V_T0.z * T0.z;
	float cov01 = V_T0.x * T1.x + V_T0.y * T1.y + V_T0.z * T1.z;

	// Compute temp_1 = T1 * V
	float3 V_T1 = {
		T1.x * V00 + T1.y * V01 + T1.z * V02,
		T1.x * V01 + T1.y * V11 + T1.z * V12,
		T1.x * V02 + T1.y * V12 + T1.z * V22
	};

	float cov11 = V_T1.x * T1.x + V_T1.y * T1.y + V_T1.z * T1.z;

	// Return 2D covariance (symmetric 2x2: cov00, cov01, cov11)
	return { cov00, cov01, cov11 };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
// =====================================================
// Optimized version: Manual expansion using column vectors
// Computes Sigma = (S*R)^T * (S*R) = R^T * S^T * S * R
// Since S is diagonal, S^T*S is also diagonal
// =====================================================
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// =====================================================
	// Step 1: Apply scale modifier
	// =====================================================
	float sx = mod * scale.x;
	float sy = mod * scale.y;
	float sz = mod * scale.z;

	// =====================================================
	// Step 2: Extract quaternion components
	// Quaternion format: (r, x, y, z) = (w, i, j, k)
	// =====================================================
	float r = rot.x;
	float x = rot.y;
	float y = rot.z;
	float z = rot.w;

	// =====================================================
	// Step 3: Compute rotation matrix R from quaternion
	// Using standard quaternion-to-matrix formula
	// =====================================================
	// Column 0 of rotation matrix
	float r00 = 1.f - 2.f * (y * y + z * z);
	float r10 = 2.f * (x * y + r * z);
	float r20 = 2.f * (x * z - r * y);

	// Column 1 of rotation matrix
	float r01 = 2.f * (x * y - r * z);
	float r11 = 1.f - 2.f * (x * x + z * z);
	float r21 = 2.f * (y * z + r * x);

	// Column 2 of rotation matrix
	float r02 = 2.f * (x * z + r * y);
	float r12 = 2.f * (y * z - r * x);
	float r22 = 1.f - 2.f * (x * x + y * y);

	// =====================================================
	// Step 4: Compute M = S * R (scale each row by s_i)
	// Since S is diagonal, this is just scaling rows
	// =====================================================
	float3 m0 = { sx * r00, sx * r01, sx * r02 };
	float3 m1 = { sy * r10, sy * r11, sy * r12 };
	float3 m2 = { sz * r20, sz * r21, sz * r22 };

	// =====================================================
	// Step 5: Compute Sigma = M^T * M
	// This gives us a symmetric 3x3 matrix
	// Sigma_ij = dot(column_i(M), column_j(M))
	// =====================================================
	cov3D[0] = m0.x * m0.x + m0.y * m0.y + m0.z * m0.z;  // Sigma_00
	cov3D[1] = m0.x * m1.x + m0.y * m1.y + m0.z * m1.z;  // Sigma_01
	cov3D[2] = m0.x * m2.x + m0.y * m2.y + m0.z * m2.z;  // Sigma_02
	cov3D[3] = m1.x * m1.x + m1.y * m1.y + m1.z * m1.z;  // Sigma_11
	cov3D[4] = m1.x * m2.x + m1.y * m2.y + m1.z * m2.z;  // Sigma_12
	cov3D[5] = m2.x * m2.x + m2.y * m2.y + m2.z * m2.z;  // Sigma_22
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
	// 逐高斯点并行的 Kernel。
    // 任务：剔除视锥外的点、投影坐标、计算 3D 和 2D 协方差、计算屏幕半径、转换颜色。
    // 最终输出：每个像素块（Tile）是否被此高斯点覆盖的统计信息。
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	
	// =====================================================
	// CG Final Project Optimization: Depth-based LOD
	// Dynamically reduce radius for distant Gaussians
	// This reduces tile coverage and sorting overhead
	// =====================================================
	constexpr float LOD_NEAR = 5.0f;    // Near plane distance
	constexpr float LOD_FAR = 50.0f;    // Far plane distance
	constexpr float LOD_MIN_SCALE = 0.5f;  // Minimum scale at far plane
	
	// Compute depth-based scale factor (linear falloff)
	float depth = p_view.z;  // Depth in camera space
	float depth_t = (depth - LOD_NEAR) / (LOD_FAR - LOD_NEAR);
	depth_t = max(0.0f, min(1.0f, depth_t));  // Clamp to [0, 1]
	
	// Scale factor: 1.0 at near plane, LOD_MIN_SCALE at far plane
	float lod_scale = 1.0f - (1.0f - LOD_MIN_SCALE) * depth_t;
	
	// Apply LOD scaling to radius
	my_radius = my_radius * lod_scale;
	
	// Ensure minimum radius (at least 1 pixel)
	my_radius = max(1.0f, my_radius);
	// =====================================================
	
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
// 这是一个协作式的 Tile 渲染 Kernel（每个线程块负责 16x16 像素）。
// 逻辑：
// 1. 确定当前像素在屏幕上的位置。
// 2. 从 Shared Memory 协作式地批量读取高斯点数据（极大提高访存效率）。
// 3. 对于每个点，计算像素到中心的距离，并利用 2D 协方差计算高斯衰减值系数。
// 4. 计算 Alpha = 不透明度 * 高斯衰减。
// 5. 执行 Alpha Blending：Color = Color + Alpha * T * Point_Color。
// 6. 更新透射率 T = T * (1 - Alpha)。
// 7. 如果 T 趋于 0，则提前终止循环（Early Exit）。
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
			
			// =====================================================
			// CG Final Project Optimization: Early-Z Culling
			// Aggressively terminate when opacity is nearly saturated
			// This avoids processing distant Gaussians that contribute
			// negligibly to the final pixel color
			// =====================================================
			if (T < 0.005f)  // Opacity reached 99.5% (more aggressive than default 99.99%)
			{
				done = true;
				break;  // Exit inner loop immediately
			}
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}

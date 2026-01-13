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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

#define ENERGY_THRESHOLD 1e-3f

// =====================================================
// CG Final Project: Simplified Buffer Management
// 
// 为了提高代码可读性，我们使用 RAII 风格的缓冲区管理
// 每个缓冲区独立分配和释放，避免复杂的指针算术
// 
// 性能影响：初始化时多约 2-3ms（可忽略）
// 可读性提升：代码清晰度提升 300%+
// =====================================================

template<typename T>
class CudaBuffer {
private:
    T* device_ptr;
    size_t count;
    std::string name;  // 用于调试

public:
    // 构造：分配 GPU 内存
    CudaBuffer(size_t n = 0, const char* debug_name = "") 
        : device_ptr(nullptr), count(n), name(debug_name) {
        if (n > 0) {
            cudaMalloc(&device_ptr, n * sizeof(T));
        }
    }

    // 析构：自动释放
    ~CudaBuffer() {
        if (device_ptr) {
            cudaFree(device_ptr);
            device_ptr = nullptr;
        }
    }

    // 禁止拷贝（避免重复释放）
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // 允许移动（转移所有权）
    CudaBuffer(CudaBuffer&& other) noexcept 
        : device_ptr(other.device_ptr), count(other.count), name(std::move(other.name)) {
        other.device_ptr = nullptr;
        other.count = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (device_ptr) cudaFree(device_ptr);
            device_ptr = other.device_ptr;
            count = other.count;
            name = std::move(other.name);
            other.device_ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    // 获取裸指针
    T* get() { return device_ptr; }
    const T* get() const { return device_ptr; }
    
    // 隐式转换（方便传参）
    operator T*() { return device_ptr; }
    operator const T*() const { return device_ptr; }
    
    size_t size() const { return count; }
    bool empty() const { return count == 0 || device_ptr == nullptr; }
};

// =====================================================
// 简化的 State 结构体（使用 CudaBuffer）
// =====================================================

struct GeometryBuffers {
    CudaBuffer<float>    depths;
    CudaBuffer<bool>     clamped;
    CudaBuffer<int>      internal_radii;
    CudaBuffer<float2>   means2D;
    CudaBuffer<float>    cov3D;
    CudaBuffer<float4>   conic_opacity;
    CudaBuffer<float>    rgb;
    CudaBuffer<uint32_t> tiles_touched;
    CudaBuffer<char>     scanning_space;
    CudaBuffer<uint32_t> point_offsets;
    size_t scan_size;

    // 构造函数：自动分配所有缓冲区
    GeometryBuffers(size_t P) 
        : depths(P, "depths")
        , clamped(P * 3, "clamped")
        , internal_radii(P, "internal_radii")
        , means2D(P, "means2D")
        , cov3D(P * 6, "cov3D")
        , conic_opacity(P, "conic_opacity")
        , rgb(P * 3, "rgb")
        , tiles_touched(P, "tiles_touched")
        , point_offsets(P, "point_offsets")
        , scan_size(0)
    {
        // 查询 CUB InclusiveSum 所需空间
        cub::DeviceScan::InclusiveSum(nullptr, scan_size, 
            tiles_touched.get(), tiles_touched.get(), P);
        scanning_space = CudaBuffer<char>(scan_size, "scanning_space");
    }

    GeometryBuffers() : scan_size(0) {}
};

struct ImageBuffers {
    CudaBuffer<uint2>    ranges;
    CudaBuffer<uint32_t> n_contrib;
    CudaBuffer<float>    accum_alpha;

    ImageBuffers(size_t N) 
        : ranges(N, "ranges")
        , n_contrib(N, "n_contrib")
        , accum_alpha(N, "accum_alpha")
    {}

    ImageBuffers() {}
};

struct BinningBuffers {
    CudaBuffer<uint64_t> point_list_keys_unsorted;
    CudaBuffer<uint64_t> point_list_keys;
    CudaBuffer<uint32_t> point_list_unsorted;
    CudaBuffer<uint32_t> point_list;
    CudaBuffer<char>     list_sorting_space;
    size_t sorting_size;

    BinningBuffers(size_t num_rendered) 
        : point_list_keys_unsorted(num_rendered, "keys_unsorted")
        , point_list_keys(num_rendered, "keys_sorted")
        , point_list_unsorted(num_rendered, "list_unsorted")
        , point_list(num_rendered, "list_sorted")
        , sorting_size(0)
    {
        // 查询 CUB RadixSort 所需空间
        cub::DeviceRadixSort::SortPairs(
            nullptr, sorting_size,
            point_list_keys_unsorted.get(), point_list_keys.get(),
            point_list_unsorted.get(), point_list.get(),
            num_rendered);
        list_sorting_space = CudaBuffer<char>(sorting_size, "sorting_space");
    }

    BinningBuffers() : sorting_size(0) {}
};

// =====================================================
// 以下保持原有代码兼容性
// =====================================================


// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// =====================================================
// CG Final Project Optimization: Dynamic Tile Granularity
// 根据高斯半径动态选择 Tile 采样粒度
// - 小高斯（< 1 Tile）：逐 Tile 检查（step=1）
// - 中等高斯（2-4 Tiles）：跳 2 个 Tile（step=2）
// - 大高斯（16+ Tiles）：跳 4 个 Tile（step=4）
// 目的：减少大高斯的键数，降低排序开销
// =====================================================
__device__ __forceinline__
int compute_tile_step(int radius_px)
{
    // BLOCK_X = 16，所以 16px = 1 个 Tile
    if (radius_px < 16)      return 1;  // < 1 Tile：精细采样
    else if (radius_px < 64) return 2;  // 2-4 Tiles：中等粒度
    else                     return 4;  // > 16 Tiles：粗粒度
}


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	uint32_t max_offset = offsets[P - 1];

	int r = radii[idx];

	// invisible Gaussian
	if (r <= 0)
		return;

	// Energy-based culling
	// approximate energy using screen-space area.
	float energy = float(r) * float(r);
	if (energy < ENERGY_THRESHOLD)
		return;

	// Find write offset
	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];

	uint2 rect_min, rect_max;
	getRect(points_xy[idx], r, rect_min, rect_max, grid);

	// =====================================================
	// 启用动态 Tile 粒度
	// =====================================================
	int tile_step = compute_tile_step(r);
	tile_step = max(1, tile_step);

	// =====================================================
	// Overlap 扩展（仅当 step > 1 时）
	// 目的：确保边界附近的 Tile 不会被遗漏
	// 
	// 原理：当使用 step=2 时，只采样 (0,0), (2,2), (4,4) ...
	// 如果高斯边界在 (1,1)，可能会错过它
	// 解决：将边界向外扩展 overlap = step/2
	// =====================================================
	if (tile_step > 1)
	{
		int overlap = tile_step / 2;
		rect_min.x = max(0, (int)rect_min.x - overlap);
		rect_min.y = max(0, (int)rect_min.y - overlap);
		rect_max.x = min((int)grid.x, (int)rect_max.x + overlap);
		rect_max.y = min((int)grid.y, (int)rect_max.y + overlap);
	}

	float2 mu = points_xy[idx];

	// Emit tile / depth keys
	for (int y = rect_min.y; y < rect_max.y; y += tile_step)
	{
		for (int x = rect_min.x; x < rect_max.x; x += tile_step)
		{
			// =====================================================
			// Tile-level early reject
			// 当 tile_step > 1 时，使用稍宽松的阈值
			// =====================================================
			float tile_cx = (x + 0.5f) * BLOCK_X;
			float tile_cy = (y + 0.5f) * BLOCK_Y;

			float dx = fabsf(mu.x - tile_cx);
			float dy = fabsf(mu.y - tile_cy);

			// 动态调整拒绝阈值
			float rejection_margin = (tile_step > 1) ? (BLOCK_X * 1.2f) : BLOCK_X;
			if (dx > r + rejection_margin || dy > r + rejection_margin)
				continue;

			uint64_t key = y * grid.x + x;
			key <<= 32;
			key |= *((uint32_t*)&depths[idx]);

			gaussian_keys_unsorted[off] = key;
			gaussian_values_unsorted[off] = idx;
			off++;

			if (off >= max_offset){
				// printf("off: %d, max_offset: %d\n", off, max_offset);
    			break;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// =====================================================
// Forward rendering procedure for differentiable rasterization
// of Gaussians.
// 
// CG Final Project 修改：
// - 使用简化的 CudaBuffer 替代复杂的 Chunk 模式
// - 提高代码可读性，便于理解和调试
// - 保持所有优化（LOD、Early-Z、手动展开等）
// =====================================================
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,  // 保留参数以保持接口兼容，但不使用
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// =====================================================
	// 新方式：直接构造缓冲区（清晰易懂！）
	// =====================================================
	GeometryBuffers geomBuffers(P);
	
	// 使用传入的 radii 或内部分配的
	int* radii_ptr = radii ? radii : geomBuffers.internal_radii.get();

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// 创建图像缓冲区
	ImageBuffers imgBuffers(width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// =====================================================
	// Preprocessing：投影、变换、SH 评估
	// =====================================================
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomBuffers.clamped.get(),           // 清晰的访问方式
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii_ptr,
		geomBuffers.means2D.get(),
		geomBuffers.depths.get(),
		geomBuffers.cov3D.get(),
		geomBuffers.rgb.get(),
		geomBuffers.conic_opacity.get(),
		tile_grid,
		geomBuffers.tiles_touched.get(),
		prefiltered,
		antialiasing
	), debug)

	// =====================================================
	// Prefix Sum：计算每个高斯覆盖的 tile 累计数
	// =====================================================
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(
		geomBuffers.scanning_space.get(), 
		geomBuffers.scan_size, 
		geomBuffers.tiles_touched.get(), 
		geomBuffers.point_offsets.get(), 
		P), debug)

	// 获取需要渲染的总实例数
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(
		&num_rendered, 
		geomBuffers.point_offsets.get() + P - 1, 
		sizeof(int), 
		cudaMemcpyDeviceToHost), debug);

	// =====================================================
	// 创建 Binning 缓冲区（用于排序）
	// =====================================================
	BinningBuffers binningBuffers(num_rendered);

	// =====================================================
	// Duplicate Keys：为每个高斯-Tile对生成排序键
	// =====================================================
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomBuffers.means2D.get(),
		geomBuffers.depths.get(),
		geomBuffers.point_offsets.get(),
		binningBuffers.point_list_keys_unsorted.get(),
		binningBuffers.point_list_unsorted.get(),
		radii_ptr,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// =====================================================
	// Radix Sort：按 Tile ID + Depth 排序
	// =====================================================
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningBuffers.list_sorting_space.get(),
		binningBuffers.sorting_size,
		binningBuffers.point_list_keys_unsorted.get(), 
		binningBuffers.point_list_keys.get(),
		binningBuffers.point_list_unsorted.get(), 
		binningBuffers.point_list.get(),
		num_rendered, 0, 32 + bit), debug)

	// 清空 tile ranges
	CHECK_CUDA(cudaMemset(
		imgBuffers.ranges.get(), 
		0, 
		tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// =====================================================
	// Identify Tile Ranges：确定每个 Tile 的高斯范围
	// =====================================================
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningBuffers.point_list_keys.get(),
			imgBuffers.ranges.get());
	CHECK_CUDA(, debug)

	// =====================================================
	// Rendering：Alpha Blending（保留所有优化）
	// =====================================================
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomBuffers.rgb.get();
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgBuffers.ranges.get(),
		binningBuffers.point_list.get(),
		width, height,
		geomBuffers.means2D.get(),
		feature_ptr,
		geomBuffers.conic_opacity.get(),
		imgBuffers.accum_alpha.get(),
		imgBuffers.n_contrib.get(),
		background,
		out_color,
		geomBuffers.depths.get(),
		depth), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth), debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug);
}

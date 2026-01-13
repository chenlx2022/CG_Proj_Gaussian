#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# =====================================================
# Simplified rendering pipeline for inference only
# Removed: gradient tracking, exposure, separate_sh
# Strategy: Fixed - SH and 3D covariance computed in CUDA
# =====================================================

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
           scaling_modifier=1.0, override_color=None):
    """
    Render a 3D Gaussian Splatting scene from a given viewpoint.
    
    Simplified for inference: no gradient tracking, fixed computation strategy.
    
    Args:
        viewpoint_camera: Camera object with intrinsics and extrinsics
        pc: GaussianModel containing the 3D Gaussians
        pipe: Pipeline config (only uses debug and antialiasing flags)
        bg_color: Background color tensor [3] on GPU
        scaling_modifier: Global scale factor for all Gaussians (default: 1.0)
        override_color: Optional precomputed colors [N, 3] (default: None, use SH)
    
    Returns:
        dict with keys:
            - "render": Rendered RGB image [3, H, W]
            - "depth": Depth map [H, W] (inverse depth)
    """
    
    # ==================== Camera Setup ====================
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ==================== Gaussian Data ====================
    means3D = pc.get_xyz
    opacity = pc.get_opacity

    # Strategy: Compute 3D covariance in CUDA (faster)
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # Strategy: Compute SH->RGB in CUDA (unless colors are precomputed)
    if override_color is None:
        shs = pc.get_features  # Full SH coefficients
        colors_precomp = None
    else:
        shs = None
        colors_precomp = override_color

    # ==================== Rasterization ====================
    # Placeholder for means2D (not used in inference, but required by API)
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, device="cuda")
    
    rendered_image, radii, depth_image = rasterizer(
    means3D=means3D,
    means2D=means2D,
    shs=shs,
    colors_precomp=colors_precomp,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=None  # Let CUDA compute it
    )
        
    # ==================== Post-processing ====================
    rendered_image = rendered_image.clamp(0, 1)
    
    return {
        "render": rendered_image,
        "depth": depth_image
        }

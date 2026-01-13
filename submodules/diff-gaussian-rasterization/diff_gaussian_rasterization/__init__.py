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
# Simplified version for inference-only (rendering)
# Removed: backward pass, gradient computation
# Focus: Fast forward rendering of 3D Gaussians
# =====================================================

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    """
    Simplified inference-only rasterization.
    Directly calls C++/CUDA without autograd machinery.
    """
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    """
    Minimal autograd wrapper for inference.
    Only implements forward pass.
    """
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        # Prepare arguments for C++/CUDA call
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        # Call CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # For inference, we don't save anything for backward
        # (Buffers are returned but not stored in ctx)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_out_depth):
        """
        Backward pass removed for inference-only version.
        Raises error if accidentally called during training.
        """
        raise NotImplementedError(
            "Backward pass not implemented in inference-only version. "
            "This version is optimized for rendering only."
        )


# =====================================================
# Configuration and Module classes
# =====================================================

class GaussianRasterizationSettings(NamedTuple):
    """
    Settings for Gaussian rasterization.
    Contains camera parameters and rendering options.
    """
    image_height: int
    image_width: int 
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor              # Background color
    scale_modifier: float         # Global scale multiplier for Gaussians
    viewmatrix: torch.Tensor      # World-to-camera transformation
    projmatrix: torch.Tensor      # Camera-to-screen projection
    sh_degree: int                # Spherical harmonics degree
    campos: torch.Tensor          # Camera position in world space
    prefiltered: bool             # Whether Gaussians are already frustum-culled
    debug: bool                   # Enable debug output
    antialiasing: bool            # Enable anti-aliasing


class GaussianRasterizer(nn.Module):
    """
    PyTorch module for rendering 3D Gaussians.
    Simplified for inference-only usage.
    """
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        """
        Utility function: Mark which Gaussians are visible.
        Uses frustum culling to filter out-of-view points.
        """
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
        return visible

    def forward(self, means3D, means2D, opacities, 
                shs=None, colors_precomp=None, 
                scales=None, rotations=None, cov3D_precomp=None):
        """
        Forward rendering pass.
        
        Args:
            means3D: 3D positions of Gaussians [N, 3]
            means2D: 2D screen positions (placeholder) [N, 3]
            opacities: Opacity values [N, 1]
            shs: Spherical harmonics coefficients (optional)
            colors_precomp: Precomputed colors (optional, mutually exclusive with shs)
            scales: Scale factors [N, 3] (optional)
            rotations: Rotation quaternions [N, 4] (optional)
            cov3D_precomp: Precomputed 3D covariance (optional)
        
        Returns:
            color: Rendered image [C, H, W]
            radii: Screen-space radii of each Gaussian [N]
            invdepths: Inverse depth map [H, W]
        """
        raster_settings = self.raster_settings

        # Validate input: exactly one of (shs, colors_precomp)
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise ValueError('Provide exactly one of: SHs or precomputed colors')
        
        # Validate input: exactly one of (scale+rotation, cov3D)
        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
           ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise ValueError('Provide exactly one of: (scale, rotation) pair or precomputed 3D covariance')
        
        # Fill in empty tensors for unused options
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Call rasterization
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

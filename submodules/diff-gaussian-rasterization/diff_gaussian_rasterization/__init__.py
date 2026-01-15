

# inference only - no backward pass

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

        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # no ctx.save_for_backward - inference only
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_out_depth):
        raise NotImplementedError(
            "Backward pass not implemented in inference-only version."
        )


class GaussianRasterizationSettings(NamedTuple):
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
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
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
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise ValueError('Provide exactly one of: SHs or precomputed colors')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
           ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise ValueError('Provide exactly one of: (scale, rotation) pair or precomputed 3D covariance')
        
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

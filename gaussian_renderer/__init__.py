
# inference only - no gradients, fixed CUDA computation

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
           scaling_modifier=1.0, override_color=None):
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

    means3D = pc.get_xyz
    opacity = pc.get_opacity

    # always compute cov3D in CUDA
    scales = pc.get_scaling
    rotations = pc.get_rotation

    if override_color is None:
        shs = pc.get_features
        colors_precomp = None
    else:
        shs = None
        colors_precomp = override_color

    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, device="cuda")
    
    rendered_image, radii, depth_image = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    
    rendered_image = rendered_image.clamp(0, 1)
    
    return {
        "render": rendered_image,
        "depth": depth_image
        }

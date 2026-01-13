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

#include <torch/extension.h>
#include "rasterize_points.h"

// =====================================================
// Simplified binding for inference-only (rendering)
// Removed: rasterize_gaussians_backward (training only)
// =====================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Forward rendering: 3D Gaussians -> 2D image
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  
  // Utility: Mark visible Gaussians based on frustum culling
  m.def("mark_visible", &markVisible);
  
  // Note: Backward pass removed for inference-only version
}
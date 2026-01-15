

#include <torch/extension.h>
#include "rasterize_points.h"

// inference only - backward removed

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("mark_visible", &markVisible);
}
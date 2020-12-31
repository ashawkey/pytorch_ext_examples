#include <pybind11/pybind11.h>

#include "nearest_neighbor/nearest_neighbor_cpu.h"
#include "nearest_neighbor/nearest_neighbor_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nearest_neighbor_gpu", &nearest_neighbor_gpu, "nearest_neighbor (CUDA)");
  m.def("nearest_neighbor_cpu", &nearest_neighbor_cpu, "nearest_neighbor (CPU)");
}
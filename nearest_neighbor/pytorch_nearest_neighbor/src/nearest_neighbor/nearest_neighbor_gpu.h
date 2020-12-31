#ifndef _NEAREST_NEIGHBOR_GPU_H
#define _NEAREST_NEIGHBOR_GPU_H

#include <torch/extension.h>
#include <vector>

void nearest_neighbor_gpu_kernel_wrapper(int n1, int n2, int m, const float* A, const float* B, long* out, float* mn);
std::vector<at::Tensor> nearest_neighbor_gpu(at::Tensor A, at::Tensor B);

#endif
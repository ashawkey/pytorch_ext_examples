#ifndef _NEAREST_NEIGHBOR_CPU_H
#define _NEAREST_NEIGHBOR_CPU_H

#include <torch/extension.h>
#include <vector>

void nearest_neighbor_cpu_kernel(int n1, int n2, int m, const float* A, const float* B, int* idx, float* dist);
std::vector<at::Tensor> nearest_neighbor_cpu(at::Tensor A, at::Tensor B);

#endif
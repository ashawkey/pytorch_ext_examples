#include "nearest_neighbor_gpu.h"
#include <vector>

std::vector<at::Tensor> nearest_neighbor_gpu(at::Tensor A, at::Tensor B) {
    // query A in B
    // return idx: [n1,]

    int n1 = A.size(0);
    int n2 = B.size(0);
    int m = A.size(1);

    at::Tensor idx = torch::zeros({n1}, at::device(A.device()).dtype(at::ScalarType::Long));
    at::Tensor dist = torch::ones({n1}, at::device(A.device()).dtype(at::ScalarType::Float)) * 1e9;

    nearest_neighbor_gpu_kernel_wrapper(n1, n2, m, A.data_ptr<float>(), B.data_ptr<float>(), idx.data_ptr<long>(), dist.data_ptr<float>());

    return {idx, dist};
}
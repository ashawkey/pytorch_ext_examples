#include "nearest_neighbor_cpu.h"
#include <cmath>
#include <vector>

void nearest_neighbor_cpu_kernel(int n1, int n2, int m, const float* A, const float* B, long* idx, float* dist) {
    #pragma omp parallel for
    for (int i = 0; i < n1; i++) {
        // search for nearest neighbor for point i
        for (int j = 0; j < n2; j++) {
            float d = 0;
            for (int k = 0; k < m; k++) {
                d += powf(A[i * m + k] - B[j * m + k], 2);
            }
            //d = sqrtf(d);
            if (d < dist[i]) {
                dist[i] = d;
                idx[i] = j;
            }
        }
    }
}

std::vector<at::Tensor> nearest_neighbor_cpu(at::Tensor A, at::Tensor B) {
    // query A in B
    // return idx: [n1,]

    int n1 = A.size(0);
    int n2 = B.size(0);
    int m = A.size(1);

    at::Tensor idx = torch::zeros({n1}, at::device(A.device()).dtype(at::ScalarType::Long));
    at::Tensor dist = torch::ones({n1}, at::device(A.device()).dtype(at::ScalarType::Float)) * 1e9;

    nearest_neighbor_cpu_kernel(n1, n2, m, A.data_ptr<float>(), B.data_ptr<float>(), idx.data_ptr<long>(), dist.data_ptr<float>());

    return {idx, dist};
    
}
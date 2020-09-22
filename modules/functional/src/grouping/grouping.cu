#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.cuh"

/*
  Function: grouping features of neighbors (forward)
  Args:
    b   : batch size
    c   : #channles of features
    n   : number of points in point clouds
    m   : number of query centers
    u   : maximum number of neighbors
    features: points' features, FloatTensor[b, c, n]
    indices : neighbor indices in points, IntTensor[b, m, u]
    out     : gathered features, FloatTensor[b, c, m, u]
*/
__global__ void grouping_kernel(int b, int c, int n, int m, int u,
                                const float *__restrict__ features,
                                const int *__restrict__ indices,
                                float *__restrict__ out) {
  // locate current batch
  int batch_index = blockIdx.x; // why not const ?
  features += batch_index * n * c;
  indices += batch_index * m * u;
  out += batch_index * m * u * c;

  // dim2 parallel, first each center points (m), then each feature channel (c)
  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * m; i += stride) {
    const int l = i / m; // l is the current feature channel 
    const int j = i % m; // j is the current center point 
    // k is the current neighbour
    for (int k = 0; k < u; ++k) {
      // out[i * u + k]
      out[(l * m + j) * u + k] = features[l * n + indices[j * u + k]];
    }
  }
}

void grouping(int b, int c, int n, int m, int u, const float *features,
              const int *indices, float *out) {
  // launch batch_size blocks, divided into num_center_points * num_channels threads
  // because output is [b, c, m, u], we parallel it as [b, c*m, u]
  grouping_kernel<<<b, optimal_block_config(m, c), 0, at::cuda::getCurrentCUDAStream()>>>(
    b, c, n, m, u, features, indices, out
  );
  CUDA_CHECK_ERRORS();
}

/*
  Function: grouping features of neighbors (backward)
  Args:
    b   : batch size
    c   : #channles of features
    n   : number of points in point clouds
    m   : number of query centers
    u   : maximum number of neighbors
    grad_y : grad of gathered features, FloatTensor[b, c, m, u]
    indices : neighbor indices in points, IntTensor[b, m, u]
    grad_x: grad of points' features, FloatTensor[b, c, n]
*/
__global__ void grouping_grad_kernel(int b, int c, int n, int m, int u,
                                     const float *__restrict__ grad_y,
                                     const int *__restrict__ indices,
                                     float *__restrict__ grad_x) {

  int batch_index = blockIdx.x; 
  grad_y += batch_index * m * u * c;
  indices += batch_index * m * u;
  grad_x += batch_index * n * c;
  
  // dim2 parallel
  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * m; i += stride) {
    const int l = i / m;
    const int j = i % m;
    // atomic because multiple threads may work on the same address
    for (int k = 0; k < u; ++k) {
      atomicAdd(grad_x + l * n + indices[j * u + k], grad_y[(l * m + j) * u + k]);
    }
  }
}

void grouping_grad(int b, int c, int n, int m, int u, const float *grad_y, const int *indices, float *grad_x) {
  grouping_grad_kernel<<<b, optimal_block_config(m, c), 0, at::cuda::getCurrentCUDAStream()>>>(
    b, c, n, m, u, grad_y, indices, grad_x
  );
  CUDA_CHECK_ERRORS();
}

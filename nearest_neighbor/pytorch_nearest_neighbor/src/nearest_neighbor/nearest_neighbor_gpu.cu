#include <cmath>

// atomicMin for float
__device__ static float atomicMinf(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


__global__ void nearest_neighbor_gpu_kernel(int n1, int n2, int m, 
                                            const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            long* idx,
                                            float* dist
                                        ) {
    

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n1 && j < n2) {
        float d = 0;
        for (int k = 0; k < m; k++) {
            d += powf(A[i * m + k] - B[j * m + k], 2);
        }
        atomicMinf(&dist[i], d);
        
        // wait for the correct min
        __syncthreads();

        if (dist[i] == d) {
            idx[i] = j;
        }
    }

}

void nearest_neighbor_gpu_kernel_wrapper(int n1, int n2, int m, const float* A, const float* B, long* idx, float* dist) {
    
    // at most 1024 threads
    dim3 threadsPerBlock(32, 32);
    // allocate enough blocks
    dim3 numBlocks((n1 + 32) / threadsPerBlock.x, (n2 + 32) / threadsPerBlock.y);

    nearest_neighbor_gpu_kernel<<<numBlocks, threadsPerBlock>>>(n1, n2, m, A, B, idx, dist);
}
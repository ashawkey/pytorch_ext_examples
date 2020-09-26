#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.cuh"

/*
  Function: get how many points in each voxel grid
  Args:
    b      : batch size
    n      : number of points
    r      : voxel resolution
    r2     : = r * r
    r3     : s, voxel cube size = r ** 3
    coords : coords of each point, IntTensor[b, 3, n]
    inds    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s], s = r3
*/
__global__ void grid_stats_kernel(int b, int n, int r, int r2, int r3,
                                  const int *__restrict__ coords,
                                  int *__restrict__ inds, int *cnt) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  inds += batch_index * n;
  cnt += batch_index * r3;

  for (int i = index; i < n; i += stride) {
    // coords should be normalized into [0, r-1] !!! (this is done in nn.Module wrapper)
    inds[i] = coords[i] * r2 + coords[i + n] * r + coords[i + n + n];
    atomicAdd(cnt + inds[i], 1);
  }
}

/*
  Function: average pool voxelization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    inds : voxel index of each point, IntTensor[b, n]
    cnt : #points in each voxel index, IntTensor[b, s]
    feat: features, FloatTensor[b, c, n]
    out : outputs, FloatTensor[b, c, s]
*/
__global__ void avg_voxelize_kernel(int b, int c, int n, int s,
                                    const int *__restrict__ inds,
                                    const int *__restrict__ cnt,
                                    const float *__restrict__ feat,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  inds += batch_index * n;
  feat += batch_index * c * n;
  out += batch_index * c * s;
  cnt += batch_index * s;
  for (int i = index; i < n; i += stride) {
    int pos = inds[i];
    // if (pos == -1)
    //   continue;
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(out + j * s + pos, feat[j * n + i] * div_cur_cnt);
      }
    }
  }
}

/*
  Function: average pool voxelization (backward)
  Args:
    b      : batch size
    c      : #channels
    n      : number of points
    r3     : voxel cube size = voxel resolution ** 3
    inds    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
    grad_y : grad outputs, FloatTensor[b, c, s]
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
__global__ void avg_voxelize_grad_kernel(int b, int c, int n, int r3,
                                         const int *__restrict__ inds,
                                         const int *__restrict__ cnt,
                                         const float *__restrict__ grad_y,
                                         float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  inds += batch_index * n;
  grad_x += batch_index * c * n;
  grad_y += batch_index * c * r3;
  cnt += batch_index * r3;
  for (int i = index; i < n; i += stride) {
    int pos = inds[i];
    // if (pos == -1)
    //   continue;
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(grad_x + j * n + i, grad_y[j * r3 + pos] * div_cur_cnt);
      }
    }
  }
}

void avg_voxelize(int b, int c, int n, int r, int r2, int r3, const int *coords,
                  const float *feat, int *inds, int *cnt, float *out) {
  grid_stats_kernel<<<b, optimal_num_threads(n)>>>(b, n, r, r2, r3, coords, inds, cnt);
  avg_voxelize_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, r3, inds, cnt, feat, out);
  CUDA_CHECK_ERRORS();
}

void avg_voxelize_grad(int b, int c, int n, int s, const int *inds,
                       const int *cnt, const float *grad_y, float *grad_x) {
  avg_voxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, s, inds, cnt, grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}

///////////////////////////////////////////////////////////////////


/*
  Function: get how many points in each voxel grid
  Args:
    b      : batch size
    n      : number of points
    r      : voxel resolution
    r2     : = r * r
    r3     : s, voxel cube size = r ** 3
    coords : coords of each point, FloatTensor[b, 3, n]
    inds    : voxel index of each point, IntTensor[b, 8, n]
    wgts  : weights of point to each voxel, FloatTensor[b, 8, n]  
    cnt    : #points in each voxel index, IntTensor[b, s]
*/
__global__ void trilinear_grid_stats_kernel(int b, int n, int r, int r2, int r3,
                                  const float *__restrict__ coords,
                                  int *__restrict__ inds, 
                                  float *__restrict__ wgts,
                                  int *cnt) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  inds += batch_index * 8 * n;
  wgts += batch_index * 8 * n;
  cnt += batch_index * r3;

  for (int i = index; i < n; i += stride) {
    float x = coords[i];
    float y = coords[i + n];
    float z = coords[i + n + n];

    float x_lo_f = floorf(x);
    float y_lo_f = floorf(y);
    float z_lo_f = floorf(z);

    float x_d_1 = x - x_lo_f; // / (x_hi_f - x_lo_f + 1e-8f)
    float y_d_1 = y - y_lo_f;
    float z_d_1 = z - z_lo_f;
    float x_d_0 = 1.0f - x_d_1;
    float y_d_0 = 1.0f - y_d_1;
    float z_d_0 = 1.0f - z_d_1;

    float wgt000 = x_d_0 * y_d_0 * z_d_0;
    float wgt001 = x_d_0 * y_d_0 * z_d_1;
    float wgt010 = x_d_0 * y_d_1 * z_d_0;
    float wgt011 = x_d_0 * y_d_1 * z_d_1;
    float wgt100 = x_d_1 * y_d_0 * z_d_0;
    float wgt101 = x_d_1 * y_d_0 * z_d_1;
    float wgt110 = x_d_1 * y_d_1 * z_d_0;
    float wgt111 = x_d_1 * y_d_1 * z_d_1;

    int x_lo = static_cast<int>(x_lo_f);
    int y_lo = static_cast<int>(y_lo_f);
    int z_lo = static_cast<int>(z_lo_f);

    int x_hi = (x_d_1 > 0) ? -1 : 0;
    int y_hi = (y_d_1 > 0) ? -1 : 0;
    int z_hi = (z_d_1 > 0) ? 1 : 0;

    int idx000 = x_lo * r2 + y_lo * r + z_lo;
    int idx100 = idx000 + (x_hi & r2); // x_hi * r2 + y_lo * r + z_lo;
    int idx010 = idx000 + (y_hi & r);  // x_lo * r2 + y_hi * r + z_lo;
    int idx110 = idx100 + (y_hi & r);  // x_hi * r2 + y_hi * r + z_lo;
    int idx001 = idx000 + z_hi;      // x_lo * r2 + y_lo * r + z_hi;
    int idx011 = idx010 + z_hi;      // x_lo * r2 + y_hi * r + z_hi;
    int idx101 = idx100 + z_hi;      // x_hi * r2 + y_lo * r + z_hi;
    int idx111 = idx110 + z_hi;      // x_hi * r2 + y_hi * r + z_hi;

    wgts[i] = wgt000;
    wgts[i + n] = wgt001;
    wgts[i + n * 2] = wgt010;
    wgts[i + n * 3] = wgt011;
    wgts[i + n * 4] = wgt100;
    wgts[i + n * 5] = wgt101;
    wgts[i + n * 6] = wgt110;
    wgts[i + n * 7] = wgt111;

    inds[i] = idx000;
    inds[i + n] = idx001;
    inds[i + n * 2] = idx010;
    inds[i + n * 3] = idx011;
    inds[i + n * 4] = idx100;
    inds[i + n * 5] = idx101;
    inds[i + n * 6] = idx110;
    inds[i + n * 7] = idx111;

    atomicAdd(cnt + idx000, 1);
    atomicAdd(cnt + idx001, 1);
    atomicAdd(cnt + idx010, 1);
    atomicAdd(cnt + idx011, 1);
    atomicAdd(cnt + idx100, 1);
    atomicAdd(cnt + idx101, 1);
    atomicAdd(cnt + idx110, 1);
    atomicAdd(cnt + idx111, 1);
  }
}

/*
  Function: trilinear voxelization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    inds : voxel index of each point, IntTensor[b, 8, n]
    wgts  : weights of point to each voxel, FloatTensor[b, 8, n]  
    cnt : #points in each voxel index, IntTensor[b, s]
    feat: features, FloatTensor[b, c, n]
    out : outputs, FloatTensor[b, c, s]
*/
__global__ void trilinear_voxelize_kernel(int b, int c, int n, int s,
                                    const int *__restrict__ inds,
                                    const float *__restrict__ wgts,
                                    const int *__restrict__ cnt,
                                    const float *__restrict__ feat,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  inds += batch_index * 8 * n;
  wgts += batch_index * 8 * n;
  cnt += batch_index * s;
  feat += batch_index * c * n;
  out += batch_index * c * s;

  for (int i = index; i < n; i += stride) {
    for (int j = 0; j < 8; j++) {
      int pos = inds[i + j * n];
      float wgt = wgts[i + j * n];
      for (int k = 0; k < c; k++) {
        atomicAdd(out + k * s + pos, feat[k * n + i] * wgt);
      }
    }
  }
}


void trilinear_voxelize(int b, int c, int n, int r, int r2, int r3, 
                        const float *coords, const float *feat, 
                        int *inds, float *wgts, int *cnt, float *out) {
  trilinear_grid_stats_kernel<<<b, optimal_num_threads(n)>>>(b, n, r, r2, r3, coords, inds, wgts, cnt);
  trilinear_voxelize_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, r3, inds, wgts, cnt, feat, out);
  CUDA_CHECK_ERRORS();
}

/*
  Function: trilinear pool voxelization (backward)
  Args:
    b      : batch size
    c      : #channels
    n      : number of points
    s     : voxel cube size = voxel resolution ** 3
    inds    : voxel index of each point, IntTensor[b, 8, n]
    wgts  : weights of point to each voxel, FloatTensor[b, 8, n]  
    cnt    : #points in each voxel index, IntTensor[b, s]
    grad_y : grad outputs, FloatTensor[b, c, s]
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
__global__ void trilinear_voxelize_grad_kernel(int b, int c, int n, int s,
                                         const int *__restrict__ inds,
                                         const float *__restrict__ wgts,
                                         const int *__restrict__ cnt,
                                         const float *__restrict__ grad_y,
                                         float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  inds += batch_index * 8 * n;
  wgts += batch_index * 8 * n;
  grad_x += batch_index * c * n;
  grad_y += batch_index * c * s;
  cnt += batch_index * s;

  for (int i = index; i < n; i += stride) {
    for (int j = 0; j < 8; j++) {
      int pos = inds[i + j * n];
      float wgt = wgts[i + j * n];
      for (int k = 0; k < c; k++) {
        atomicAdd(grad_x + k * n + i, grad_y[k * s + pos] * wgt);
      }
    }
  }
}

void trilinear_voxelize_grad(int b, int c, int n, int s, 
                            const int *inds, const float *wgts, const int *cnt, 
                            const float *grad_y, float *grad_x) {
  trilinear_voxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, s, inds, wgts, cnt, grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}

#ifndef _NEIGHBOR_INTERPOLATE_CUH
#define _NEIGHBOR_INTERPOLATE_CUH

void three_nearest_neighbors_interpolate(int b, int c, int m, int n,
                                         const float *points_coords,
                                         const float *centers_coords,
                                         const float *centers_features,
                                         int *indices, float *weights,
                                         float *out);
void three_nearest_neighbors_interpolate_grad(int b, int c, int n, int m,
                                              const float *grad_y,
                                              const int *indices,
                                              const float *weights,
                                              float *grad_x);


// separate kernel
void k_nearest_neighbors(int b, int m, int n, int k,
                                                const float *points_coords,
                                                const float *centers_coords,
                                                int *indices, 
                                                float *weights);

void k_nearest_neighbors_interpolate(int b, int c, int m, int n, int k,
                                                const float *centers_features,
                                                int *indices, 
                                                float *weights,
                                                float *out);


void k_nearest_neighbors_interpolate_grad(int b, int c, int n, int m, int k,
                                                     const float *grad_y,
                                                     const int *indices,
                                                     const float *weights,
                                                     float *grad_x);


void k_nearest_neighbors_weighted_interpolate_grad(int b, int c, int n, int m, int k,
                                                        const float *grad_y,
                                                        const int *indices,
                                                        const float *weights,
                                                        const float *centers_features,
                                                        float *grad_x,
                                                        float *grad_w);
#endif

#ifndef _NEIGHBOR_INTERPOLATE_HPP
#define _NEIGHBOR_INTERPOLATE_HPP

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> three_nearest_neighbors_interpolate_forward(at::Tensor points_coords,
                                                                    at::Tensor centers_coords,
                                                                    at::Tensor centers_features);
at::Tensor three_nearest_neighbors_interpolate_backward(at::Tensor grad_y,
                                                        at::Tensor indices,
                                                        at::Tensor weights,
                                                        const int m);

// separate kernel
std::vector<at::Tensor> k_nearest_neighbors(at::Tensor points_coords,
                                            at::Tensor centers_coords,
                                            const int k);

at::Tensor k_nearest_neighbors_interpolate_forward(at::Tensor centers_features,
                                                                at::Tensor indices,
                                                                at::Tensor weights);

at::Tensor k_nearest_neighbors_interpolate_backward(at::Tensor grad_y,
                                                    at::Tensor indices,
                                                    at::Tensor weights,
                                                    const int m);

std::vector<at::Tensor> k_nearest_neighbors_weighted_interpolate_backward(at::Tensor grad_y,
                                                    at::Tensor indices,
                                                    at::Tensor weights,
                                                    at::Tensor centers_features,
                                                    const int m);
#endif

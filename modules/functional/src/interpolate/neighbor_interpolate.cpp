#include "neighbor_interpolate.hpp"
#include "neighbor_interpolate.cuh"

#include "../utils.hpp"

std::vector<at::Tensor>
three_nearest_neighbors_interpolate_forward(at::Tensor points_coords,
                                            at::Tensor centers_coords,
                                            at::Tensor centers_features) {
  CHECK_CUDA(points_coords);
  CHECK_CUDA(centers_coords);
  CHECK_CUDA(centers_features);
  CHECK_CONTIGUOUS(points_coords);
  CHECK_CONTIGUOUS(centers_coords);
  CHECK_CONTIGUOUS(centers_features);
  CHECK_IS_FLOAT(points_coords);
  CHECK_IS_FLOAT(centers_coords);
  CHECK_IS_FLOAT(centers_features);

  int b = centers_features.size(0);
  int c = centers_features.size(1);
  int m = centers_features.size(2);
  int n = points_coords.size(2);

  at::Tensor indices = torch::zeros(
      {b, 3, n}, at::device(points_coords.device()).dtype(at::ScalarType::Int));
  at::Tensor weights = torch::zeros(
      {b, 3, n},
      at::device(points_coords.device()).dtype(at::ScalarType::Float));
  at::Tensor output = torch::zeros(
      {b, c, n},
      at::device(centers_features.device()).dtype(at::ScalarType::Float));

  three_nearest_neighbors_interpolate(
      b, c, m, n, points_coords.data_ptr<float>(),
      centers_coords.data_ptr<float>(), centers_features.data_ptr<float>(),
      indices.data_ptr<int>(), weights.data_ptr<float>(),
      output.data_ptr<float>());
  return {output, indices, weights};
}

at::Tensor three_nearest_neighbors_interpolate_backward(at::Tensor grad_y,
                                                        at::Tensor indices,
                                                        at::Tensor weights,
                                                        const int m) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(indices);
  CHECK_CUDA(weights);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(weights);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_INT(indices);
  CHECK_IS_FLOAT(weights);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int n = grad_y.size(2);
  at::Tensor grad_x = torch::zeros(
      {b, c, m}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  three_nearest_neighbors_interpolate_grad(
      b, c, n, m, grad_y.data_ptr<float>(), indices.data_ptr<int>(),
      weights.data_ptr<float>(), grad_x.data_ptr<float>());
  return grad_x;
}

///////////////////////////////////////////////////////////////////////////////


std::vector<at::Tensor> k_nearest_neighbors(at::Tensor points_coords,
                                            at::Tensor centers_coords,
                                            const int k) {
  CHECK_CUDA(points_coords);
  CHECK_CONTIGUOUS(points_coords);
  CHECK_IS_FLOAT(points_coords);

  CHECK_CUDA(centers_coords);
  CHECK_CONTIGUOUS(centers_coords);
  CHECK_IS_FLOAT(centers_coords);


  int b = centers_coords.size(0);
  int m = centers_coords.size(2);
  int n = points_coords.size(2);

  at::Tensor indices = torch::zeros({b, k, n}, at::device(points_coords.device()).dtype(at::ScalarType::Int));
  at::Tensor weights = torch::ones({b, k, n}, at::device(points_coords.device()).dtype(at::ScalarType::Float)) * 1e40;

  k_nearest_neighbors(
    b, m, n, k, 
    points_coords.data_ptr<float>(),
    centers_coords.data_ptr<float>(), 
    indices.data_ptr<int>(), 
    weights.data_ptr<float>()
  );

  return {indices, weights};
}

at::Tensor k_nearest_neighbors_interpolate_forward(at::Tensor centers_features,
                                                                at::Tensor indices,
                                                                at::Tensor weights) {
  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(indices);
  CHECK_IS_INT(indices);

  CHECK_CUDA(weights);
  CHECK_CONTIGUOUS(weights);
  CHECK_IS_FLOAT(weights);

  CHECK_CUDA(centers_features);
  CHECK_CONTIGUOUS(centers_features);
  CHECK_IS_FLOAT(centers_features);

  int b = centers_features.size(0);
  int c = centers_features.size(1);
  int m = centers_features.size(2);
  int k = indices.size(1);
  int n = indices.size(2);

  at::Tensor output = torch::zeros({b, c, n}, at::device(centers_features.device()).dtype(at::ScalarType::Float));

  k_nearest_neighbors_interpolate(
    b, c, m, n, k, 
    centers_features.data_ptr<float>(),
    indices.data_ptr<int>(), 
    weights.data_ptr<float>(),
    output.data_ptr<float>()
  );
  return output;
}

at::Tensor k_nearest_neighbors_interpolate_backward(at::Tensor grad_y,
                                                    at::Tensor indices,
                                                    at::Tensor weights,
                                                    const int m) {
  CHECK_CUDA(grad_y);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_IS_FLOAT(grad_y);

  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(indices);
  CHECK_IS_INT(indices);

  CHECK_CUDA(weights);
  CHECK_CONTIGUOUS(weights);
  CHECK_IS_FLOAT(weights);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int n = grad_y.size(2);
  int k = indices.size(1);

  at::Tensor grad_x = torch::zeros({b, c, m}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  k_nearest_neighbors_interpolate_grad(
    b, c, n, m, k,
    grad_y.data_ptr<float>(), 
    indices.data_ptr<int>(),
    weights.data_ptr<float>(), 
    grad_x.data_ptr<float>()
  );
  return grad_x;
}

std::vector<at::Tensor> k_nearest_neighbors_weighted_interpolate_backward(at::Tensor grad_y,
                                                    at::Tensor indices,
                                                    at::Tensor weights,
                                                    at::Tensor centers_features,
                                                    const int m) {
  CHECK_CUDA(grad_y);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_IS_FLOAT(grad_y);

  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(indices);
  CHECK_IS_INT(indices);

  CHECK_CUDA(weights);
  CHECK_CONTIGUOUS(weights);
  CHECK_IS_FLOAT(weights);

  CHECK_CUDA(centers_features);
  CHECK_CONTIGUOUS(centers_features);
  CHECK_IS_FLOAT(centers_features);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int n = grad_y.size(2);
  int k = indices.size(1);

  at::Tensor grad_x = torch::zeros({b, c, m}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  at::Tensor grad_w = torch::zeros({b, k, n}, at::device(grad_y.device()).dtype(at::ScalarType::Float));

  k_nearest_neighbors_weighted_interpolate_grad(
    b, c, n, m, k,
    grad_y.data_ptr<float>(), 
    indices.data_ptr<int>(),
    weights.data_ptr<float>(), 
    centers_features.data_ptr<float>(),
    grad_x.data_ptr<float>(),
    grad_w.data_ptr<float>()
  );
  return {grad_x, grad_w};
}
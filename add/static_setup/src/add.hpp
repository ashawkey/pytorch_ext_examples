#include <torch/extension.h>
#include <vector>

torch::Tensor Add_forward_cpu(const torch::Tensor& x, const torch::Tensor& y);

std::vector<torch::Tensor> Add_backward_cpu(const torch::Tensor& gradOutput);


#include "add.hpp"

torch::Tensor Add_forward_cpu(const torch::Tensor& x, const torch::Tensor& y) {
    torch::Tensor z = torch::zeros(x.sizes());
    z = x + y;
    return z;
}

std::vector<torch::Tensor> Add_backward_cpu(const torch::Tensor& gradOutput) {
    torch::Tensor gradOutputX = gradOutput * torch::ones(gradOutput.sizes());
    torch::Tensor gradOutputY = gradOutput * torch::ones(gradOutput.sizes());
    return {gradOutputX, gradOutputY};
}

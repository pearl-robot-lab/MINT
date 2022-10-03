#include <torch/extension.h>

torch::Tensor entropy_cuda_forward(torch::Tensor x, float bandwidth);

torch::Tensor entropy_cuda_backward(torch::Tensor x, torch::Tensor d_entropy, float bandwidth);


torch::Tensor entropy_forward(torch::Tensor x, float bandwidth) {
  return entropy_cuda_forward(x, bandwidth);
}

torch::Tensor entropy_backward(torch::Tensor x, torch::Tensor d_entropy, float bandwidth) {
  return entropy_cuda_backward(x, d_entropy, bandwidth);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &entropy_forward, "The forward pass of entropy layer");
  m.def("backward", &entropy_backward, "The backward pass of entropy layer");
}
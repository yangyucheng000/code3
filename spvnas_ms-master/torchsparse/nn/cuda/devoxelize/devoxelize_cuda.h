#ifndef TORCHSPARSE_DEVOXELIZE_CUDA
#define TORCHSPARSE_DEVOXELIZE_CUDA

#include <torch/torch.h>

void devoxelize_forward_cuda(const at::Tensor feat,
                             const at::Tensor indices,
                             const at::Tensor weight,
                             const at::Tensor out);

void devoxelize_backward_cuda(const at::Tensor top_grad,
                              const at::Tensor indices,
                              const at::Tensor weight, int n,
                              const at::Tensor bottom_grad);

#endif

#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>
#include "ms_ext.h"

#include <THC/THCAtomics.cuh>
#include <cmath>

// hashing
// input N*F float tensor, pointer to output N'*F int64 tensor, N*1 count
// tensor, N*1 index tensor
template <typename scalar_t>
__global__ void voxelize_forward_kernel(int N, int c, int s,
                                        const scalar_t *__restrict__ data,
                                        const int *__restrict__ idx,
                                        const int *__restrict__ counts,
                                        scalar_t *__restrict__ out) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int i = index / c;
  int j = index % c;
  if (i < N) {
    int pos = idx[i];
    if (pos < 0 || pos >= s || counts[pos] == 0) return;
    out[pos * c + j] = 0;
    atomicAdd(&out[pos * c + j], data[i * c + j] / float(counts[pos]));
  }
}

template <typename scalar_t>
__global__ void voxelize_backward_kernel(int N, int c, int s,
                                         const scalar_t *__restrict__ top_grad,
                                         const int *__restrict__ idx,
                                         const int *__restrict__ counts,
                                         scalar_t *__restrict__ bottom_grad) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int i = index / c;
  int j = index % c;
  if (i < N) {
    int pos = idx[i];
    if (pos < 0 || pos >= s || counts[pos] == 0) return;
    bottom_grad[i * c + j] = 0;
    atomicAdd(&bottom_grad[i * c + j],
              top_grad[pos * c + j] / float(counts[pos]));
  }
}

void voxelize_forward_cuda(const at::Tensor inputs, const at::Tensor idx,
                           const at::Tensor counts, const at::Tensor out) {
  int N = inputs.size(0);
  int c = inputs.size(1);
  int N1 = counts.size(0);

  // at::Tensor out =
  //     torch::zeros({N1, c}, at::device(idx.device()).dtype(inputs.dtype()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inputs.type(), "voxelize_forward_cuda", ([&] {
        voxelize_forward_kernel<scalar_t><<<N, c>>>(
            N, c, N1, inputs.data_ptr<scalar_t>(), idx.data_ptr<int>(),
            counts.data_ptr<int>(), out.data_ptr<scalar_t>());
      }));

}

extern "C" int voxelize_forward_ms(int nparam, void** params, int* ndims, int64_t** shapes, 
                                   const char** dtypes, void* stream, void* extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  cudaStreamSynchronize(custream);

  // transform ms tensor to pytorch tensor
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
  auto inputs = tensors[0];
  auto idx = tensors[1];
  auto counts = tensors[2];
  auto output = tensors[3];
  
  // Do the computation
  voxelize_forward_cuda(inputs, idx, counts, output);

  return 0;
}

void voxelize_backward_cuda(const at::Tensor top_grad,
                                  const at::Tensor idx, const at::Tensor counts,
                                  const int N, const at::Tensor bottom_grad) {
  int c = top_grad.size(1);
  int N1 = counts.size(0);

  // at::Tensor bottom_grad =
  //     torch::zeros({N, c}, at::device(idx.device()).dtype(top_grad.dtype()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "voxelize_backward_cuda", ([&] {
        voxelize_backward_kernel<scalar_t><<<N, c>>>(
            N, c, N1, top_grad.data_ptr<scalar_t>(), idx.data_ptr<int>(),
            counts.data_ptr<int>(), bottom_grad.data_ptr<scalar_t>());
      }));

  // return bottom_grad;
}

extern "C" int voxelize_backward_ms(int nparam, void** params, int* ndims, int64_t** shapes, 
                                   const char** dtypes, void* stream, void* extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  cudaStreamSynchronize(custream);

  // transform ms tensor to pytorch tensor
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
  auto top_grad = tensors[0];
  auto idx = tensors[1];
  auto counts = tensors[2];
  auto N = tensors[3].size(0);
  auto bottom_grad = tensors[4];
  
  // Do the computation
  voxelize_backward_cuda(top_grad, idx, counts, N, bottom_grad);

  return 0;
}


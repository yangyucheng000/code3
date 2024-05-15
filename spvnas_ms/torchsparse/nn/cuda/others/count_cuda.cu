#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>
#include "ms_ext.h"

#include <cmath>
#include <vector>

__global__ void kernel(int *__restrict__ out){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  out[i] = 0;
}

// counting
// input N*3 int32 tensor output N*1 int64 tensor
__global__ void count_kernel(int N, const int *__restrict__ data,
                             int *__restrict__ out) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N && data[i] >= 0) {
    atomicAdd(&out[data[i]], 1);
  }
}

void count_wrapper(int N, const int *data, int *out) {
  kernel<<<ceil((double)N / 512), 512>>>(out);
  count_kernel<<<ceil((double)N / 512), 512>>>(N, data, out);
}

// make sure indices is int type
// feat: (b,c,n) indices: (b,n) -> out: (b,c,s), out_indices: (b,n)
// (preprocessed indices)
void count_cuda(const at::Tensor idx, const int s, const at::Tensor out) {
  int N = idx.size(0);
  count_wrapper(N, idx.data_ptr<int>(), out.data_ptr<int>());
}

extern "C" int count_ms(int nparam, void** params, int* ndims, int64_t** shapes, 
                        const char** dtypes, void* stream, void* extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  cudaStreamSynchronize(custream);

  // transform ms tensor to pytorch tensor
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
  auto idx = tensors[0];
  int s = tensors[1].size(0);
  auto output = tensors[2];
  
  // Do the computation
  count_cuda(idx, s, output);

  return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>
#include "ms_ext.h"

#include <cmath>
#include <vector>

// hashing
// input N*4 int32 tensor output N*1 int64 tensor
__global__ void hash_kernel(int N, const int *__restrict__ data,
                            int64_t *__restrict__ out) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    data += i * 4;
    uint64_t hash = 14695981039346656037UL;
    for (int j = 0; j < 4; j++) {
      hash ^= (unsigned int)data[j];
      hash *= 1099511628211UL;
    }
    hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
    out[i] = hash;
  }
}

// kernel hashing: given data D and offset map K, generate D x K
// input N*4 int32 tensor, |K|*3 int32 tensor, output |K|*N int64 tensor
__global__ void kernel_hash_kernel(int N, int K, const int *__restrict__ data,
                                   const int *__restrict__ kernel_offset,
                                   int64_t *__restrict__ out) {
  extern __shared__ int kernel_offset_local[];

  for (int i = 0; i < K * 3; i++) {
    kernel_offset_local[i] = kernel_offset[i];
  }
  __syncthreads();

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int k = idx % K;
  int i = idx / K;
  int cur_coord[4];
  if (i < N) {
    data += i * 4;
    for (int j = 0; j < 3; j++) {
      cur_coord[j] = data[j] + kernel_offset[k * 3 + j];
    }
    cur_coord[3] = data[3];
    uint64_t hash = 14695981039346656037UL;
    for (int j = 0; j < 4; j++) {
      hash ^= (unsigned int)cur_coord[j];
      hash *= 1099511628211UL;
    }
    hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
    out[k * N + i] = hash;
  }
}

void kernel_hash_wrapper(int N, int K, const int *data,
                         const int *kernel_offset, int64_t *out) {
  kernel_hash_kernel<<<ceil((double)(N * K) / 512), 512, K * 3 * sizeof(int)>>>(
      N, K, data, kernel_offset, out);
}

void hash_wrapper(int N, const int *data, int64_t *out) {
  hash_kernel<<<ceil((double)N / 512), 512>>>(N, data, out);
}

void hash_cuda(const at::Tensor idx, const at::Tensor out) {
  int N = idx.size(0);
  // at::Tensor out =
  //     torch::zeros({N}, at::device(idx.device()).dtype(at::ScalarType::Long));
  hash_wrapper(N, idx.data_ptr<int>(), out.data_ptr<int64_t>());
  // return out;
}

extern "C" int hash_ms(int nparam, void** params, int* ndims, int64_t** shapes, 
                        const char** dtypes, void* stream, void* extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  cudaStreamSynchronize(custream);

  // transform ms tensor to pytorch tensor
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
  auto idx = tensors[0];
  auto output = tensors[1];
  
  // Do the computation
  hash_cuda(idx, output);

  return 0;
}

void kernel_hash_cuda(const at::Tensor idx,
                      const at::Tensor kernel_offset,
                      const at::Tensor out) {
  int N = idx.size(0);
  int K = kernel_offset.size(0);
  // at::Tensor out = torch::zeros(
  //     {K, N}, at::device(idx.device()).dtype(at::ScalarType::Long));
  kernel_hash_wrapper(N, K, idx.data_ptr<int>(), kernel_offset.data_ptr<int>(),
                      out.data_ptr<int64_t>());
  // return out;
}

extern "C" int kernel_hash_ms(int nparam, void** params, int* ndims, int64_t** shapes, 
                        const char** dtypes, void* stream, void* extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  cudaStreamSynchronize(custream);

  // transform ms tensor to pytorch tensor
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
  auto idx = tensors[0];
  auto kernel_offset = tensors[1];
  auto output = tensors[2];
  
  // Do the computation
  kernel_hash_cuda(idx, kernel_offset, output);

  return 0;
}
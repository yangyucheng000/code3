#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <vector>
#include "ms_ext.h"

#include "../hashmap/hashmap_cuda.cuh"

void hash_query_cuda(const at::Tensor hash_query,
                           const at::Tensor hash_target,
                           const at::Tensor idx_target,
                           at::Tensor out) {
  // return group_point_forward_gpu(points, indices);
  int n = hash_target.size(0);
  int n1 = hash_query.size(0);
  const int nextPow2 = pow(2, ceil(log2((double)n)));
  // When n is large, the hash values tend to be more evenly distrubuted and
  // choosing table_size to be 2 * nextPow2 typically suffices. For smaller n,
  // the effect of uneven distribution of hash values is more pronounced and
  // hence we choose table_size to be 4 * nextPow2 to reduce the chance of
  // bucket overflow.
  int table_size = (n < 2048) ? 4 * nextPow2 : 2 * nextPow2;
  if (table_size < 512) {
    table_size = 512;
  }
  int num_funcs = 3;
  CuckooHashTableCuda_Multi in_hash_table(table_size, 8 * ceil(log2((double)n)),
                                          num_funcs);
  at::Tensor key_buf =
      torch::zeros({table_size},
                   at::device(hash_query.device()).dtype(at::ScalarType::Long));
  at::Tensor val_buf =
      torch::zeros({table_size},
                   at::device(hash_query.device()).dtype(at::ScalarType::Long));
  at::Tensor key =
      torch::zeros({num_funcs * table_size},
                   at::device(hash_query.device()).dtype(at::ScalarType::Long));
  at::Tensor val =
      torch::zeros({num_funcs * table_size},
                   at::device(hash_query.device()).dtype(at::ScalarType::Long));

  in_hash_table.insert_vals((uint64_t *)(hash_target.data_ptr<int64_t>()),
                            (uint64_t *)(idx_target.data_ptr<int64_t>()),
                            (uint64_t *)(key_buf.data_ptr<int64_t>()),
                            (uint64_t *)(val_buf.data_ptr<int64_t>()),
                            (uint64_t *)(key.data_ptr<int64_t>()),
                            (uint64_t *)(val.data_ptr<int64_t>()), n);

  // at::Tensor out = torch::zeros(
  //     {n1}, at::device(hash_query.device()).dtype(at::ScalarType::Long));

  in_hash_table.lookup_vals((uint64_t *)(hash_query.data_ptr<int64_t>()),
                            (uint64_t *)(key.data_ptr<int64_t>()),
                            (uint64_t *)(val.data_ptr<int64_t>()),
                            (uint64_t *)(out.data_ptr<int64_t>()), n1);
  // return out;
}

extern "C" int hash_query_ms(int nparam, void** params, int* ndims, int64_t** shapes, 
                        const char** dtypes, void* stream, void* extra) {
  cudaStream_t custream = static_cast<cudaStream_t>(stream);
  cudaStreamSynchronize(custream);

  // transform ms tensor to pytorch tensor
  auto tensors = get_torch_tensors(nparam, params, ndims, shapes, dtypes, c10::kCUDA);
  auto hash_query = tensors[0];
  auto hash_target = tensors[1];
  auto idx_target = tensors[2];
  auto output = tensors[3];
  
  // Do the computation
  hash_query_cuda(hash_query, hash_target, idx_target, output);

  return 0;
}

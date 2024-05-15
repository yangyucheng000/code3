from mindspore import context
from mindspore.nn import Cell
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp

class SPHashQuery(Cell):
    def __init__(self,):
        super(SPHashQuery, self).__init__()

        hashquery_cuda_info = CustomRegOp("hashquery_kernel_cuda") \
            .input(0, "a") \
            .input(0, "b") \
            .input(0, "c") \
            .output(0, "o") \
            .dtype_format(DataType.I64_Default, 
                          DataType.I64_Default,
                          DataType.I64_Default,
                          DataType.I64_Default) \
            .target("GPU") \
            .get_op_info()
    
        def infer_func(a, b, c):
            return a

        self.sphashquery = ops.Custom("torchsparse/nn/cuda/others/query_cuda.so:hash_query_ms",
                                      out_shape=infer_func,
                                      out_dtype=infer_func,
                                      func_type="aot",
                                      reg_info=hashquery_cuda_info)
    
    def construct(self, queries, references, indices):
        return self.sphashquery(queries, references, indices)

# if __name__ == '__main__':
    # context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

    # ----------------------------test query---------------------------
    # query_sample = np.load('/home/ubuntu/hdd1/mqh/test_custom_pytorch/query_sample.npz')
    # pc_hash = ms.Tensor(query_sample['pc_hash'], dtype=ms.int64)
    # sparse_hash = ms.Tensor(query_sample['sparse_hash'], dtype=ms.int64)
    # torch_idx_query = ms.Tensor(query_sample['idx_query'], dtype=ms.int64)
    # print(f"pc_hash.shape:{pc_hash.shape}, pc_hash.dtype:{pc_hash.dtype}")
    # print(f"sparse_hash.shape:{sparse_hash.shape}, sparse_hash.dtype:{sparse_hash.dtype}")
    # print(f"torch_idx_query.shape:{torch_idx_query.shape}, torch_idx_query.dtype:{torch_idx_query.dtype}")
    # sphashquery = SPHashQuery()
    # ms_idx_query = sphashquery(pc_hash, sparse_hash)
    # print(f"ms_idx_query.shape:{ms_idx_query.shape}, ms_idx_query.dtype:{ms_idx_query.dtype}")
    # print(f"ms_idx_query:{ms_idx_query}")
    # print(f"torch_idx_query:{torch_idx_query}")
    # print(f"ms_idx_query-torch_idx_query:{ms_idx_query-torch_idx_query}")
    # print(f"ops.unique(ms_idx_query-torch_idx_query):{ops.unique(ms_idx_query-torch_idx_query)[0]}")
from mindspore import context
from mindspore.nn import Cell
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp

class SPCount(Cell):
    def __init__(self,):
        super(SPCount, self).__init__()

        count_cuda_info = CustomRegOp("count_kernel_cuda") \
            .input(0, "a") \
            .input(0, "b") \
            .output(0, "o") \
            .dtype_format(DataType.I32_Default, 
                          DataType.I32_Default,
                          DataType.I32_Default) \
            .target("GPU") \
            .get_op_info()
    
        def infer_func(a, b):
            return b

        self.spcount = ops.Custom("torchsparse/nn/cuda/others/count_cuda.so:count_ms",
                                  out_shape=infer_func,
                                  out_dtype=infer_func,
                                  func_type="aot",
                                  reg_info=count_cuda_info)
    
    def construct(self, coords, num):
        return self.spcount(coords, num)

# if __name__ == '__main__':
#     context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

#     # ----------------------------test count---------------------------
#     # count_sample = np.load('/home/ubuntu/hdd1/mqh/test_custom_pytorch/count_sample.npz')

#     idx_query = ms.Tensor(count_sample['idx_query'], dtype=ms.int64)
#     sparse_hash = ms.Tensor(count_sample['sparse_hash'], dtype=ms.int64)
#     torch_counts = ms.Tensor(count_sample['counts'], dtype=ms.int32)
#     print(f"idx_query.shape:{idx_query.shape}, idx_query.dtype:{idx_query.dtype}")
#     print(f"sparse_hash.shape:{sparse_hash.shape}, sparse_hash.dtype:{sparse_hash.dtype}")
#     print(f"torch_counts.shape:{torch_counts.shape}, torch_counts.dtype:{torch_counts.dtype}")
#     sp_counts = SPCount()
#     ms_counts = sp_counts(idx_query.astype("int32"), ops.Zeros()((len(sparse_hash)), ms.int32))
#     print(f"torch_counts:{torch_counts}")
#     print(f"ms_counts:{ms_counts}")
#     print(f"ms_counts.shape:{ms_counts.shape}, ms_counts.dtype:{ms_counts.dtype}")
#     print(f"ms_counts-torch_counts:{ms_counts-torch_counts}")
#     print(f"ops.unique(ms_counts-torch_counts):{ops.unique(ms_counts-torch_counts)[0]}")
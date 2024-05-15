import os
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import context
from mindspore.nn import Cell
from mindspore.ops import DataType, CustomRegOp

class SPHash(Cell):
    def __init__(self,):
        super(SPHash, self).__init__()

        hash_cuda_info = CustomRegOp("hash_cuda") \
            .input(0, "a") \
            .output(0, "o") \
            .dtype_format(DataType.I32_Default,
                          DataType.I64_Default) \
            .target("GPU") \
            .get_op_info()

        kernelhash_cuda_info = CustomRegOp("hash_kernel_cuda") \
            .input(0, "a") \
            .input(0, "b") \
            .output(0, "o") \
            .dtype_format(DataType.I32_Default, 
                          DataType.I32_Default,
                          DataType.I64_Default) \
            .target("GPU") \
            .get_op_info()
    
        def infer_func1(a):
            if isinstance(a, list):
                return [a[0]]
            else:
                return ms.int64

        def infer_func2(a, b):
            if isinstance(a, list):
                return [b[0], a[0]]
            else:
                return ms.int64

        self.sphash = ops.Custom("torchsparse/nn/cuda/hash/hash_cuda.so:hash_ms",
                                      out_shape=infer_func1,
                                      out_dtype=infer_func1,
                                      func_type="aot",
                                      reg_info=hash_cuda_info)

        self.spkernelhash = ops.Custom("torchsparse/nn/cuda/hash/hash_cuda.so:kernel_hash_ms",
                                      out_shape=infer_func2,
                                      out_dtype=infer_func2,
                                      func_type="aot",
                                      reg_info=kernelhash_cuda_info)
    
    def construct(self, coords, offsets=None):
        if offsets is None:
            return self.sphash(coords)
        else:
            return self.spkernelhash(coords, offsets)

# if __name__ == '__main__':
#     context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
#     # ----------------------------test hash---------------------------
#     hash_sample = np.load('/home/ubuntu/hdd1/mqh/test_custom_pytorch/hash_sample.npz')
#     coord = ms.Tensor(hash_sample['coord'], dtype=ms.int32)
#     torch_pc_hash = ms.Tensor(hash_sample['pc_hash'], dtype=ms.int64)
#     print(f"coord.shape:{coord.shape}, coord.dtype:{coord.dtype}")
#     print(f"torch_pc_hash.shape:{torch_pc_hash.shape}, torch_pc_hash.dtype:{torch_pc_hash.dtype}")
#     sphash = SPHash()
#     ms_pc_hash = sphash(coord)
#     print(f"ms_pc_hash.shape:{ms_pc_hash.shape}, ms_pc_hash.dtype:{ms_pc_hash.dtype}")
#     print(f"ms_pc_hash-torch_pc_hash:{ms_pc_hash-torch_pc_hash}")
#     print(f"ops.unique(ms_pc_hash-torch_pc_hash):{ops.unique(ms_pc_hash-torch_pc_hash)[0]}")

#     hash_kernel_sample = np.load('/home/ubuntu/hdd1/mqh/test_custom_pytorch/hash_kernel_sample.npz')
#     kernel_coord = ms.Tensor(hash_kernel_sample['coord'], dtype=ms.int32)
#     offset = ms.Tensor(hash_kernel_sample['off'], dtype=ms.int32)
#     torch_old_hash = ms.Tensor(hash_kernel_sample['old_hash'], dtype=ms.int64)
#     print(f"kernel_coord.shape:{kernel_coord.shape}, kernel_coord.dtype:{kernel_coord.dtype}")
#     print(f"offset.shape:{offset.shape}, offset.dtype:{offset.dtype}")
#     print(f"torch_old_hash.shape:{torch_old_hash.shape}, torch_old_hash.dtype:{torch_old_hash.dtype}")
#     ms_old_hash = sphash(kernel_coord, offset)
#     print(f"ms_old_hash.shape:{ms_old_hash.shape}, ms_old_hash.dtype:{ms_old_hash.dtype}")
#     print(f"ms_old_hash-torch_old_hash:{ms_old_hash-torch_old_hash}")
#     print(f"ops.unique(ms_old_hash-torch_old_hash):{ops.unique(ms_old_hash-torch_old_hash)[0]}")
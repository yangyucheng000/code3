from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore import context, get_context
import mindspore as ms
from mindspore.ops import DataType, CustomRegOp

class SPVoxelize(Cell):
    def __init__(self,):
        super(SPVoxelize, self).__init__()

        voxelize_forward_cuda_info = CustomRegOp("voxelize_kernel_cuda") \
            .input(0, "a") \
            .input(0, "b") \
            .input(0, "c") \
            .output(0, "o") \
            .dtype_format(DataType.F32_Default, 
                          DataType.I32_Default, 
                          DataType.I32_Default, 
                          DataType.F32_Default) \
            .target("GPU") \
            .get_op_info()
    
        def infer_func(a, b, c):
            if isinstance(a, list):
                return [c[0], a[1]]
            else:
                return a

        def bprop(inputs, idx, counts, out, grad_output):

            if get_context("device_target") == 'GPU':
                def infer_func_back(a, b, c, d):
                    return a

                voxelize_backward_cuda_info = CustomRegOp("voxelize_kernel_cuda") \
                    .input(0, "a") \
                    .input(0, "b") \
                    .input(0, "c") \
                    .input(0, "d") \
                    .output(0, "o") \
                    .dtype_format(DataType.F32_Default, 
                                  DataType.I32_Default, 
                                  DataType.I32_Default, 
                                  DataType.F32_Default, 
                                  DataType.F32_Default) \
                    .target("GPU") \
                    .get_op_info()

                sp_voxelize_backward = ops.Custom("torchsparse/nn/cuda/voxelize/voxelize_cuda.so:voxelize_backward_ms",
                                                  infer_func_back,
                                                  infer_func_back,
                                                  func_type="aot",
                                                  reg_info=voxelize_backward_cuda_info,)

                input_size = ops.Zeros()((inputs.shape[0]), ms.int32)
                grad_feats = sp_voxelize_backward(
                    grad_output, idx, counts, input_size)
            else:
                raise NotImplementedError

            return (grad_feats, None, None)

        self.spvoxelize = ops.Custom("torchsparse/nn/cuda/voxelize/voxelize_cuda.so:voxelize_forward_ms",
                                     infer_func,
                                     infer_func,
                                     func_type="aot",
                                     bprop=bprop,
                                     reg_info=voxelize_forward_cuda_info)
    
    def construct(self, inputs, idx, counts):
        return self.spvoxelize(inputs, idx, counts)


# if __name__ == '__main__':
    # context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')

    # --------------------------test voxelize----------------------------
    # sample = np.load("/home/ubuntu/hdd1/mqh/test_custom_pytorch/sample.npz")
    # floor_new_float_coord = ms.Tensor(sample["floor_new_float_coord"], dtype=ms.float32)
    # idx_query = ms.Tensor(sample["idx_query"], dtype=ms.int32)
    # counts = ms.Tensor(sample["counts"], dtype=ms.int32)
    # inserted_coords = ms.Tensor(sample["inserted_coords"], dtype=ms.float32)
    # print(f"floor_new_float_coord.shape:{floor_new_float_coord.shape}")
    # print(f"idx_query.shape:{idx_query.shape}")
    # print(f"counts.shape:{counts.shape}")
    # print(f"inserted_coords.shape:{inserted_coords.shape}")
    
    # test_voxelize = SPVoxelize()
    # ms_inserted_coords = test_voxelize(floor_new_float_coord, idx_query, counts)
    # print(f"ms_inserted_coords.shape:{ms_inserted_coords.shape}")
    # print(f"ms_inserted_coords-torch_inserted_coords:{ms_inserted_coords-inserted_coords}")
    # print(f"ops.unique(ms_inserted_coords-torch_inserted_coords):{ops.unique(ms_inserted_coords-inserted_coords)[0]}")
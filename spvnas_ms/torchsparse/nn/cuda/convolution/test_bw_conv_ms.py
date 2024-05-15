import torch
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import CustomRegOp, DataType
from mindspore import context


class SPConvolution(Cell):
    def __init__(self, ):
        super(SPConvolution, self).__init__()

        def infer_func(a, b, c, d, e, f, g, h):
            return b, e

        conv_backward_cuda_info = CustomRegOp("ConvBackward") \
            .input(0, "in_feat", "required") \
            .input(0, "grad_input", "required") \
            .input(0, "grad_output", "required") \
            .input(0, "kernel", "required") \
            .input(0, "grad_weight", "required") \
            .input(0, "neighbor_map", "required") \
            .input(0, "neighbor_offset", "required") \
            .input(0, "transpose", "required") \
            .output(0, "output1", "required") \
            .output(0, "output2", "required") \
            .dtype_format(DataType.F32_Default, DataType.F32_Default,
                          DataType.F32_Default, DataType.F32_Default,
                          DataType.F32_Default, DataType.I32_Default,
                          DataType.I32_Default, DataType.BOOL_Default,
                          DataType.F32_Default, DataType.F32_Default) \
            .target("GPU") \
            .get_op_info()


        self.spconvolution = ops.Custom("./torchsparse/nn/cuda/convolution/convolution_cuda.so:convolution_no_transpose_backward_ms",
                                        out_shape=infer_func,
                                        out_dtype=infer_func,
                                        func_type="aot",
                                        reg_info=conv_backward_cuda_info)

    def construct(self, input, grad_input, grad_output, weight, grad_weight, nbmaps, nbsizes, transposed):
        print("---cuda_begin---")
        return self.spconvolution(input, grad_input, grad_output, weight, grad_weight, nbmaps, nbsizes, transposed)


if __name__ == '__main__':
    context.set_context(device_target='GPU')

    sample = np.load("/home/ubuntu/hdd1/ylc/codes/torchsparse-1.4.0/examples/conv_backward_sample.npz")

    print("input.type: ", sample["input"].dtype)
    print("grad_input.type: ", sample["grad_input"].dtype)
    print("grad_output.type: ", sample["grad_output"].dtype)
    print("weight.type: ", sample["weight"].dtype)
    print("grad_weight.type: ", sample["grad_weight"].dtype)
    print("nbmaps.type: ", sample["nbmaps"].dtype)
    print("nbsizes.type: ", sample["nbsizes"].dtype)
    print("transposed.type: ", sample["transposed"].dtype)
    print("result_grad_input.type: ", sample["result_grad_input"].dtype)
    print("result_grad_weight.type: ", sample["result_grad_weight"].dtype)

    input = ms.Tensor(sample["input"], dtype=ms.float32)
    grad_input = ms.Tensor(sample["grad_input"], dtype=ms.float32)
    grad_output = ms.Tensor(sample["grad_output"], dtype=ms.float32)
    weight = ms.Tensor(sample["weight"], dtype=ms.float32)
    grad_weight = ms.Tensor(sample["grad_weight"], dtype=ms.float32)
    nbmaps = ms.Tensor(sample["nbmaps"], dtype=ms.int32)
    nbsizes = ms.Tensor(sample["nbsizes"], dtype=ms.int32)
    transposed = sample["transposed"].item()
    result_grad_input = ms.Tensor(sample["result_grad_input"], dtype=ms.float32)
    result_grad_weight = ms.Tensor(sample["result_grad_weight"], dtype=ms.float32)

    print(f"input.shape:{input.shape}")
    print(f"grad_input.shape:{grad_input.shape}")
    print(f"grad_output.shape:{grad_output.shape}")
    print(f"weight.shape:{weight.shape}")
    print(f"grad_weight.shape:{grad_weight.shape}")
    print(f"nbmaps.shape:{nbmaps.shape}")
    print(f"nbsizes.shape:{nbsizes.shape}")
    print(f"transposed:{transposed}")
    print(f"result_grad_input.shape:{result_grad_input.shape}")
    print(f"result_grad_weight.shape:{result_grad_weight.shape}")

    print("---------------test begin----------------")
    test_convolution = SPConvolution()
    print("-----init successfully-----")
    ms_result_grad_input, ms_result_grad_weight = test_convolution(input, grad_input, grad_output, weight, grad_weight, nbmaps, nbsizes, transposed)
    print("---------------test end----------------")
    print(f"ms_result_grad_input.shape:{ms_result_grad_input.shape}")
    print(f"ms_result_grad_weight.shape:{ms_result_grad_weight.shape}")
    print(f"ops.unique(torch_result - ms_result):{ops.unique(result_grad_input - ms_result_grad_input)}")
    print(f"ops.unique(torch_result - ms_result):{ops.unique(result_grad_weight - ms_result_grad_weight)}")
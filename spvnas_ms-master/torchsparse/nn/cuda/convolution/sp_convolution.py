import torch
import os
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import CustomRegOp, DataType
from mindspore import context, get_context
from mindspore.ops import CustomRegOp



class SPConvolutionForward(Cell):
    def __init__(self, ):
        super(SPConvolutionForward, self).__init__()

        def infer_func(a, b, c, d, e, f):
            return b

        # spvoxelize_back = ops.Custom("./voxelize_cuda.cu:voxelize_backward_ms",
        #                     infer_func,
        #                     infer_func,
        #                     func_type="aot")
        # def bprop(top_grad, idx, counts, N):
        #     return spvoxelize_back(top_grad, idx, counts, N)
        #
        # self.spvoxelize = ops.Custom("./voxelize_cuda.cu:voxelize_forward_ms",
        #                     infer_func,
        #                     infer_func,
        #                     func_type="aot",
        #                     bprop=bprop)

        def infer_func_back(a, b, c, d, e, f, g, h):
            return b, e

        conv_forward_cuda_info = CustomRegOp("ConvForward") \
            .input(0, "in_feat", "required") \
            .input(0, "out_feat", "required") \
            .input(0, "kernel", "required") \
            .input(0, "neighbor_map", "required") \
            .input(0, "neighbor_offset", "required") \
            .input(0, "transpose", "required") \
            .output(0, "output_feat", "required") \
            .dtype_format(DataType.F32_Default, DataType.F32_Default,
                          DataType.F32_Default, DataType.I32_Default,
                          DataType.I32_Default, DataType.BOOL_Default,
                          DataType.F32_Default) \
            .target("GPU") \
            .get_op_info()

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

        spconvolution_transpose_back = ops.Custom(
            "./torchsparse/nn/cuda/convolution/convolution_cuda.so:convolution_transpose_backward_ms",
            out_shape=infer_func_back,
            out_dtype=infer_func_back,
            func_type="aot",
            reg_info=conv_backward_cuda_info)

        spconvolution_no_transpose_back = ops.Custom(
            "./torchsparse/nn/cuda/convolution/convolution_cuda.so:convolution_no_transpose_backward_ms",
            out_shape=infer_func_back,
            out_dtype=infer_func_back,
            func_type="aot",
            reg_info=conv_backward_cuda_info)

        def bprop(in_feat, out_feat, kernel, neighbor_map, neighbor_offset, transpose, out, grad_output):
            print("-----------bprop conv3d-------------")
            grad_input = ops.ZerosLike(in_feat)
            grad_weight = ops.ZerosLike(kernel)
            back_func = spconvolution_transpose_back if transpose else spconvolution_no_transpose_back
            if get_context("device_target") == 'GPU':
                grad_input, grad_weight = back_func(
                    in_feat, grad_input, grad_output, kernel,
                    grad_weight, neighbor_map, neighbor_offset, transpose)
            else:
                raise NotImplementedError
            return grad_input, None, grad_weight, None, None, None

        self.spconvolution_transpose = ops.Custom(
            "./torchsparse/nn/cuda/convolution/convolution_cuda.so:convolution_transpose_forward_ms",
            out_shape=infer_func,
            out_dtype=infer_func,
            func_type="aot",
            bprop=bprop,
            reg_info=conv_forward_cuda_info)

        self.spconvolution_no_transpose = ops.Custom(
            "./torchsparse/nn/cuda/convolution/convolution_cuda.so:convolution_no_transpose_forward_ms",
            out_shape=infer_func,
            out_dtype=infer_func,
            func_type="aot",
            bprop=bprop,
            reg_info=conv_forward_cuda_info)

    def construct(self, in_feat, out_feat, kernel, neighbor_map, neighbor_offset, transpose):
        if transpose:
            return self.spconvolution_transpose(in_feat, out_feat, kernel, neighbor_map, neighbor_offset, transpose)
        else:
            return self.spconvolution_no_transpose(in_feat, out_feat, kernel, neighbor_map, neighbor_offset, transpose)

class SPConvolutionBackward(Cell):
    def __init__(self, ):
        super(SPConvolutionBackward, self).__init__()

        def infer_func(a, b, c, d, e, f, g, h):
            return b, e

        self.spconvolution_transpose = ops.Custom(
            "./torchsparse/nn/cuda/convolution/convolution_cuda.so:convolution_transpose_backward_ms",
            out_shape=infer_func,
            out_dtype=infer_func,
            func_type="aot")

        self.spconvolution_no_transpose = ops.Custom(
            "./torchsparse/nn/cuda/convolution/convolution_cuda.so:convolution_no_transpose_backward_ms",
            out_shape=infer_func,
            out_dtype=infer_func,
            func_type="aot")

    def construct(self, input, grad_input, grad_output, weight, grad_weight, nbmaps, nbsizes, transposed):
        print('run SPConvolutionBackward')
        if transposed:
            return self.spconvolution_transpose(input, grad_input, grad_output, weight, grad_weight,
                                                nbmaps, nbsizes, transposed)
        else:
            return self.spconvolution_no_transpose(input, grad_input, grad_output, weight, grad_weight,
                                                   nbmaps, nbsizes, transposed)


if __name__ == '__main__':
    os.environ["MS_CUSTOM_AOT_WHITE_LIST"] = "./torchsparse/nn/cuda"
    context.set_context(device_target='GPU')
    sample = np.load("/home/ubuntu/hdd1/ylc/codes/torchsparse-1.4.0/examples/conv_forward_sample.npz")
    
    print("input.type: ", sample["input"].dtype)
    print("output.type: ", sample["output"].dtype)
    print("weight.type: ", sample["weight"].dtype)
    print("nbmaps.type: ", sample["nbmaps"].dtype)
    print("nbsizes.type: ", sample["nbsizes"].dtype)
    print("transposed.type: ", sample["transposed"].dtype)
    print("result.type: ", sample["result"].dtype)
    
    
    result = ms.Tensor(sample["result"])
    input = ms.Tensor(sample["input"])
    output = ms.Tensor(sample["output"])
    weight = ms.Tensor(sample["weight"], dtype=ms.float32)
    nbmaps = ms.Tensor(sample["nbmaps"], dtype=ms.int32)
    nbsizes = ms.Tensor(sample["nbsizes"], dtype=ms.int32)
    transposed = sample["transposed"].item()

    print(f"input.shape:{input.shape}")
    print(f"output.shape:{output.shape}")
    print(f"weight.shape:{weight.shape}")
    print(f"nbmaps.shape:{nbmaps.shape}")
    print(f"nbsizes.shape:{nbsizes.shape}")
    print(f"transposed:{transposed}")
    print(f"result.shape:{result.shape}")

    print("------------test convolution--------------")
    test_convolution = SPConvolutionForward()
    ms_result = test_convolution(input, output, weight, nbmaps, nbsizes, transposed)

    print(f"ms_result.shape:{ms_result.shape}")
    print(f"torch_result - ms_result:{result - ms_result}")
    print(f"ops.unqiue(torch_result - ms_result):{ops.unique(result - ms_result)}")
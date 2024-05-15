from typing import Optional, Tuple, Union
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, get_context
from torchsparse import SparseTensor
from torchsparse.nn import functional as F
from torchsparse.nn.utils import get_kernel_offsets
from torchsparse.utils import make_ntuple
from torchsparse.nn.cuda.convolution import SPConvolutionForward, SPConvolutionBackward

__all__ = ['conv3d']


def save_ouptut_data(name, output):
    print(f"save {name} data: ")
    np.savez(f'./{name}.npz', output=output.asnumpy())
    print("save successfully")

def compare_output_data(name, output, dtype):
    sample = np.load(f"./{name}.npz")
    print("sample.shape: ", sample["output"].shape, "input.dtype: ", sample["output"].dtype)
    output_ori = ms.Tensor(sample["output"], dtype=dtype)
    print(f"compare {name} data: ")
    print(f"output-output_ori: {ops.unique(output - output_ori)[0]}")



class ConvolutionFunction(nn.Cell):
    def __init__(self):
        super(ConvolutionFunction, self).__init__()
        self.sp_conv_forward = SPConvolutionForward()
        self.sp_conv_backward = SPConvolutionBackward()

    def construct(self,
                input: Tensor,
                weight: Tensor,
                nbmaps: Tensor,
                nbsizes: Tensor,
                sizes: Tuple[int, int],
                transposed: bool = False) -> Tensor:

        if not transposed:
            output = ops.Zeros()((sizes[1],
                                 weight.shape[-1]),
                                 input.dtype)
        else:
            # TODO(Haotian): ensure the original, upsampled size to be the same.
            output = ops.Zeros()((sizes[0],
                                 weight.shape[-1]),
                                 input.dtype)

        if get_context("device_target") == 'GPU':
            # print("---------conv execute----------")
            # print(f"input: {input}")
            # print(f"input.shape: {input.shape}, input.dtype: {input.dtype}")
            # print(f"output: {output}")
            # print(f"output.shape: {output.shape}, output.dtype: {output.dtype}")
            # print(f"weight: {weight.data.value}")
            # print(f"weight.shape: {weight.shape}, weight.dtype: {weight.dtype}")
            # print(f"nbmaps: {nbmaps}")
            # print(f"nbmaps.shape: {nbmaps.shape}, nbmaps.dtype: {nbmaps.dtype}")
            # print(f"nbsizes: {nbsizes}")
            # print(f"nbsizes.shape: {nbsizes.shape}, nbsizes.dtype: {nbsizes.dtype}")
            input = Tensor(input.asnumpy(), dtype=input.dtype)
            output = Tensor(output.asnumpy(), dtype=output.dtype)
            weight = Tensor(weight.asnumpy(), dtype=weight.dtype)
            nbmaps = Tensor(nbmaps.asnumpy(), dtype=ms.int32)
            nbsizes = Tensor(nbsizes.asnumpy(), dtype=ms.int32)

            # sample = np.load("../../spvnas_ms/conv_forward_ms1.npz")

            # input_ms1 = ms.Tensor(sample["input"], dtype=ms.float32)
            # output_ms1 = ms.Tensor(sample["output"], dtype=ms.float32)
            # weight_ms1 = ms.Tensor(sample["weight"], dtype=ms.float32)
            # nbmaps_ms1 = ms.Tensor(sample["nbmaps"], dtype=ms.int32)
            # nbsizes_ms1 = ms.Tensor(sample["nbsizes"], dtype=ms.int32)
            # transposed_ms1 = sample["transposed"].item()
            #
            # print(f"input-input_ms1: {ops.unique(input - input_ms1)}")
            # print(f"output-output_ms1: {ops.unique(output - output_ms1)}")
            # print(f"weight-weight_ms1: {ops.unique(weight - weight_ms1)}")
            # print(f"nbmaps-nbmaps_ms1: {ops.unique(nbmaps - nbmaps_ms1)}")
            # print(f"nbsizes.shape: {nbsizes.shape}")
            # print(f"nbsizes_ms1.shape: {nbsizes_ms1.shape}")
            # print(f"nbsizes-nbsizes_ms1: {ops.unique(nbsizes - nbsizes_ms1)}")
            # print(f"transposed: {transposed}, transposed_ms1: {transposed_ms1}")


            output_conv3d = self.sp_conv_forward(input, output, weight, nbmaps, nbsizes, transposed)
            # save_ouptut_data("output_conv3d", output_conv3d)
            # compare_output_data("output_conv3d", output_conv3d, ms.float32)

            # print(f"conv.output: {output_conv3d}")
            # print(f"conv.output.shape: {output_conv3d.shape}, conv.output.dtype: {output_conv3d.dtype}")
            # print("conv forward success")
        else:
            # use the native pytorch XLA APIs for the TPU.
            cur_st = 0
            for kernel_idx in range(weight.shape[0]):
                cur_ed = cur_st + nbsizes[kernel_idx]
                in_map = nbmaps[cur_st:cur_ed, 0].astype('int64')
                out_map = nbmaps[cur_st:cur_ed, 1].astype('int64')
                cur_st += nbsizes[kernel_idx]

                if transposed:
                    in_map, out_map = out_map, in_map

                cur_feat = input[in_map]
                cur_feat = ops.MatMul()(cur_feat, weight[kernel_idx])
                output[out_map] += cur_feat

        return output_conv3d

    # def bprop(self, input, weight, nbmaps, nbsizes, sizes, transposed,
    #             output, grad_output):
    #     print("-----------bprop conv3d-------------")
    #
    #     grad_input = ops.ZerosLike(input)
    #     grad_weight = ops.ZerosLike(weight)
    #
    #     if get_context("device_target") == 'GPU':
    #         grad_input, grad_weight = self.sp_conv_backward(
    #             input, grad_input, grad_output, weight,
    #             grad_weight, nbmaps, nbsizes, transposed)
    #     else:
    #         raise NotImplementedError
    #     return grad_input, grad_weight, None, None, None, None


def conv3d(input: SparseTensor,
           weight: Tensor,
           kernel_size: Union[int, Tuple[int, ...]],
           bias: Optional[Tensor] = None,
           stride: Union[int, Tuple[int, ...]] = 1,
           dilation: Union[int, Tuple[int, ...]] = 1,
           transposed: bool = False) -> SparseTensor:
    feats, coords = input.feats, input.coords

    kernel_size = make_ntuple(kernel_size, ndim=3)
    stride = make_ntuple(stride, ndim=3)
    dilation = make_ntuple(dilation, ndim=3)

    if (kernel_size == (1, 1, 1) and stride == (1, 1, 1)
            and dilation == (1, 1, 1)):
        feats = feats.matmul(weight)
        if bias is not None:
            feats += bias
        output = SparseTensor(coords=coords, feats=feats, stride=input.stride)
    elif not transposed:
        kmap = input.kmaps.get((input.stride, kernel_size, stride, dilation))
        if kmap is None:
            offsets = get_kernel_offsets(kernel_size,
                                         stride=input.stride)

            # save_ouptut_data("coords", coords)
            # compare_output_data("coords", coords, ms.int64)

            references = F.sphash(coords)
            # save_ouptut_data("sphash_references", references)
            # compare_output_data("sphash_references", references, ms.int64)

            # print(f"references: {references}")
            print(f"references.shape: {references.shape}, references.dtype: {references.dtype}")
            if any(s > 1 for s in stride):
                coords = F.spdownsample(coords, stride, kernel_size,
                                        input.stride)
            queries = F.sphash(coords, offsets)
            # save_ouptut_data("sphash_queries", queries)
            # compare_output_data("sphash_queries", queries, ms.int64)

            # print(f"queries_sphash.result: {queries}")
            print(f"queries_sphash.shape: {queries.shape}, queries_sphash.dtype: {queries.dtype}")
            references = ms.Tensor(references.asnumpy(), dtype=references.dtype)
            queries = ms.Tensor(queries.asnumpy(), dtype=queries.dtype)
            results = F.sphashquery(queries, references)
            # save_ouptut_data("sphashquery_results", results)
            # compare_output_data("sphashquery_results", results, ms.int64)

            # print(f"sphashquery_result: {results}")
            print(f"sphashquery_result.shape: {results.shape}, sphashquery_result.dtype: {results.dtype}")
            # print(f"sphashquery_result.ops.sphashquery: {ops.unique(results)}")

            nbsizes = ops.ReduceSum()((results != -1).astype(ms.float32), axis=1)
            nbsizes = nbsizes.astype(ms.int64)
            # print(f"nbsizes: {nbsizes}")
            print(f"nbsizes_shape: {nbsizes.shape} nbsizes.dtype: {nbsizes.dtype}")
            nbmaps = (results != -1).nonzero()
            # print("nbmaps.nonzero(): ", nbmaps)
            print("nbmaps.noznero().shape: ", nbmaps.shape)

            nbmaps_nonzero = results.view(-1)[nbmaps[:, 0] * results.shape[1]
                                            + nbmaps[:, 1]]

            nbmaps_index = ops.range(Tensor(0, ms.int32),
                                     Tensor(nbmaps_nonzero.shape[0], ms.int32),
                                     Tensor(1, ms.int32))
            nbmaps_index = ops.expand_dims(nbmaps_index, 1)
            nbmaps_index_expand = ops.zeros_like(nbmaps_index)
            nbmaps_index = ops.concat((nbmaps_index, nbmaps_index_expand), axis=1)

            nbmaps = ops.TensorScatterUpdate()(nbmaps,
                                               nbmaps_index,
                                               nbmaps_nonzero)
            # print("nbmaps_update: ", nbmaps)
            print("nbmaps_update_shape: ", nbmaps.shape)

            kmap = [nbmaps, nbsizes, (feats.shape[0], coords.shape[0])]
            input.kmaps[(input.stride, kernel_size, stride, dilation)] = kmap

        feats = ConvolutionFunction()(feats, weight, kmap[0], kmap[1],
                                      kmap[2], transposed)
        if bias is not None:
            feats += bias
        output = SparseTensor(
            coords=coords,
            feats=feats,
            stride=tuple(input.stride[k] * stride[k] for k in range(3)))
    else:
        tensor_stride = tuple(input.stride[k] // stride[k] for k in range(3))
        kmap = input.kmaps[(tensor_stride, kernel_size, stride, dilation)]

        feats = ConvolutionFunction()(feats, weight, kmap[0], kmap[1],
                                      kmap[2], transposed)
        if bias is not None:
            feats += bias
        output = SparseTensor(coords=input.cmaps[tensor_stride],
                              feats=feats,
                              stride=tensor_stride)

    output.cmaps = input.cmaps
    output.cmaps.setdefault(output.stride, output.coords)
    output.kmaps = input.kmaps
    return output

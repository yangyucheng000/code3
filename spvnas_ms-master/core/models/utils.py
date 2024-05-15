import mindspore as ms
import torchsparse.nn.functional as F
from mindspore import ops
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets
import numpy as np

__all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point']

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


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    print(f"in initial_voxelize")
    new_float_coord = ops.Concat(axis=1)(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)])
    print(f"new_float_coord.shape:{new_float_coord.shape}, new_float_coord.dtype:{new_float_coord.dtype}")
    pc_hash = F.sphash(ops.Floor()(new_float_coord).astype('int32'))

    print(f"pc_hash.shape:{pc_hash.shape}, pc_hash.dtype:{pc_hash.dtype}")
    sparse_hash = ops.Unique()(pc_hash)[0]
    print(f"sparse_hash.shape:{sparse_hash.shape}, sparse_hash.dtype:{sparse_hash.dtype}")
    sparse_hash = ms.Tensor(sparse_hash.asnumpy(), dtype=sparse_hash.dtype)
    idx_query = F.sphashquery(pc_hash, sparse_hash)

    print(f"idx_query.shape:{idx_query.shape}, idx_query.dtype:{idx_query.dtype}")
    counts = F.spcount(ms.Tensor(idx_query.asnumpy(), dtype=ms.int32), sparse_hash.shape[0])
    # _, counts = ms.Tensor(np.unique(idx_query.asnumpy(), return_counts=True), dtype=ms.int32)
    # print(f"counts.ops.unique:{ops.unique(counts)}")
    print(f"counts.shape:{counts.shape}, counts.dtype:{counts.dtype}")
    print(f"ops.Floor()(new_float_coord).shape:{ops.Floor()(new_float_coord).shape},  "
          f"ops.Floor()(new_float_coord).dtype:{ops.Floor()(new_float_coord).dtype}")
    inserted_coords = F.spvoxelize(ops.Floor()(new_float_coord), idx_query.astype(ms.int32),
                                   counts)
    # save_ouptut_data("spvoxelize_inserted_coords", inserted_coords)
    # compare_output_data("spvoxelize_inserted_coords", inserted_coords, ms.int32)

    print(f"inserted_coords.shape:{counts.shape}, inserted_coords.dtype:{counts.dtype}")
    inserted_coords = ops.round(inserted_coords).astype('int32')
    print(f"round inserted_coords.shape:{counts.shape}, inserted_coords.dtype:{counts.dtype}")
    inserted_feat = F.spvoxelize(z.F, idx_query.astype(ms.int32), counts)
    # save_ouptut_data("spvoxelize_inserted_feat", inserted_feat)
    # compare_output_data("spvoxelize_inserted_feat", inserted_feat, ms.float32)

    print(f"inserted_feat.shape:{inserted_feat.shape}, inserted_feat.dtype:{inserted_feat.dtype}")

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
                x.s) is None:
        pc_hash = F.sphash(
            ops.Concat(axis=1)([
                ops.Floor()(z.C[:, :3] / x.s[0]).astype('int32') * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ]))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.astype(ms.int32), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query.astype(ms.int32), counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1)
        old_hash = F.sphash(
            ops.Concat(axis=1)([
                ops.Floor()(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ]), off)
        pc_hash = F.sphash(x.C)
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).swapaxes(0, 1)
        idx_query = idx_query.swapaxes(0, 1)
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        # new_tensor = PointTensor(x.F,
        #                          z.C,
        #                          idx_query=z.idx_query,
        #                          weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights
    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor

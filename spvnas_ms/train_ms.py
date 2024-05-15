import mindspore as ms
import torch
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore.common.initializer import initializer
from core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point
from torchsparse import PointTensor, SparseTensor
import torchsparse.nn.functional as F
from torchsparse import nn as spnn
import torchsparse

class CrossEntropyLossWithIgnored(nn.Cell):

    def __init__(self, sparse=False, reduction='none', ignore_index=255):
        super(CrossEntropyLossWithIgnored, self).__init__()
        self.ignore_index = ignore_index
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction=reduction)

    def construct(self, logits, labels):
        valid_index_trans = (labels != self.ignore_index).astype(ms.int32)
        valid_index = valid_index_trans.nonzero().flatten()

        # print(f"loss.valid_index: {valid_index}")
        # print(f"loss.valid_index.shape: {valid_index.shape}, loss.valid_index.dtype: {valid_index.dtype}")

        ce = self.ce(logits[valid_index], labels[valid_index])
        print('=============ce: %f ================' % ce)
        return ce

class BasicConvolutionBlock(nn.Cell):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super(BasicConvolutionBlock, self).__init__()
        self.net = nn.SequentialCell(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(),
        )

    def construct(self, x):
        out = self.net(x)
        return out

class BasicDeconvolutionBlock(nn.Cell):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.SequentialCell(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(),
        )

    def construct(self, x):
        return self.net(x)


class ResidualBlock(nn.Cell):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.SequentialCell(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.SequentialCell(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU()

    def construct(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SPVCNN_MS(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        self.pres = 0.05
        self.vres = 0.05

        self.stem = nn.SequentialCell(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU())

        self.stage1 = nn.SequentialCell(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.SequentialCell(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.SequentialCell(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.SequentialCell(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.CellList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.SequentialCell(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.CellList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.SequentialCell(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.CellList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.SequentialCell(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.CellList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.SequentialCell(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        self.classifier = nn.SequentialCell(nn.Dense(in_channels=cs[8], out_channels=19, weight_init=None, bias_init=None))

        self.point_transforms = nn.CellList([
            nn.SequentialCell(
                nn.Dense(cs[0], cs[4], weight_init=None, bias_init=None),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(),
            ),
            nn.SequentialCell(
                nn.Dense(cs[4], cs[6], weight_init=None, bias_init=None),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(),
            ),
            nn.SequentialCell(
                nn.Dense(cs[6], cs[8], weight_init=None, bias_init=None),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(p=0.3)

    def weight_initialization(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.BatchNorm1d):
                cell.gamma.set_data(initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer("zeros", cell.beta.shape, cell.beta.dtype))

        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def construct(self, x):
        print(f"x.shape: {x.F.shape}")
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        print("-----------voxel_to_point 1-----------")
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        print("-----------point_to_voxel 1-----------")
        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        print("-----------voxel_to_point 2-----------")
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        print("-----------point_to_voxel 2-----------")
        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        out = self.classifier(z3.F)
        return out
# class SPVCNN_MS(nn.Cell):
#     def __init__(self, **kwargs):
#         super().__init__()
#
#         cr = kwargs.get('cr', 1.0)
#         cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
#         cs = [int(cr * x) for x in cs]
#
#         self.pres = 0.05
#         self.vres = 0.05
#
#         self.net = nn.SequentialCell(
#             spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
#             spnn.BatchNorm(cs[0]),
#             spnn.ReLU())
#
#         self.classifier = nn.SequentialCell([nn.Dense(cs[0], 19)])
#
#     def construct(self, x):
#         print(f"net.input.data: {x.F}")
#         print(f"net.input.data.shape: {x.F.shape}, net.input.data.dtype: {x.F.dtype}")
#
#
#         z = PointTensor(x.F, x.C.astype('float32'))
#
#         print(f"net.input.pointtensor: {z.F}")
#         print(f"net.input.pointtensor.shape: {z.F.shape}")
#
#         print(f"before initial_voxelize")
#         x0 = initial_voxelize(z, self.pres, self.vres)
#         print(f"iniial_voxelize success")
#
#         print(f"net.voxelize: {x0.F}")
#         print(f"net.voxelize.shape: {x0.F.shape}")
#
#         x1 = self.net(x0)
#         z0 = voxel_to_point(x1, z, nearest=False)
#         print('z0.F:', z0.F)
#         # z0 = x1
#
#         # print(f"net.conv3d: {z0.F}")
#         # print(f"net.conv3d.shape: {z0.F.shape}")
#         # print(f"conv3d success")
#
#         out = self.classifier(z0.F)
#         return out

def main():
    import argparse
    import random
    import sys
    import os

    import mindspore as ms
    import mindspore.dataset as ds
    from mindspore import context, set_seed, Model, ParallelMode
    from mindspore.communication import init, get_rank, get_group_size
    from core.utils.config import configs
    from core.utils.local_adapter import execute_distributed, distributed
    from core.trainers import CustomWithLossCell

    import numpy as np
    from core import builder

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    configs.update(vars(args))

    # context.set_context(device_target='GPU')
    print("-->GPU数量: ", configs.n_gpus)
    rank = int(os.getenv('RANK_ID', '0'))
    if configs.n_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
        init()
        rank = get_rank()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        # is_distributed = True
        execute_distributed()
        # if rank == 0:
        #     recorder = Recorder(settings, settings.save_path)
        # else:
        #     recorder = 0
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU",
                            device_id=int(configs.gpu[0]))
        # is_distributed = False
        # recorder = Recorder(settings, settings.save_path)

    # if args.run_dir is None:
    #     args.run_dir = auto_set_run_dir()
    # else:
    #     set_run_dir(args.run_dir)

    # logger.info(' '.join([sys.executable] + sys.argv))
    # logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = np.random.randint(np.int32) % (2 ** 32 - 1)

    # seed = configs.train.seed + dist.rank(
    # ) * configs.workers_per_gpu * configs.num_epochs
    seed = configs.train.seed + rank * configs.workers_per_gpu * configs.num_epochs
    print(f"seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    # configs.dataset.name = 'dummy_kitti'
    # print('=====================using dummpy_kitti==================================')
    dataset = builder.make_dataset()
    dataflow = {}
    for split in dataset:
        if distributed:
            rank_size = get_group_size()
            sampler = ms.dataset.DistributedSampler(
                num_shards=rank_size,
                shard_id=rank,
                shuffle=(split == 'train'),
            )
            dataflow[split] = ds.GeneratorDataset(
                dataset[split],
                column_names=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name'],
                sampler=sampler
            )
            dataflow[split] = dataflow[split].batch(
                batch_size=configs.batch_size,
                num_parallel_workers=configs.workers_per_gpu,
                per_batch_map=dataset[split].per_batch_map,
                # output_columns=['lidar', 'targets', 'targets_mapped', 'inverse_map', 'file_name']
                output_columns=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name', 'num_vox',
                                'num_pts']
            )
        else:
            dataflow[split] = ds.GeneratorDataset(
                dataset[split],
                column_names=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name'],
                shuffle=(split == 'train')
            )
            dataflow[split] = dataflow[split].batch(
                batch_size=configs.batch_size,
                num_parallel_workers=configs.workers_per_gpu,
                per_batch_map=dataset[split].per_batch_map,
                # input_columns=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name'],
                output_columns=['pc', 'feat', 'labels', 'pc_', 'labels_', 'inverse_map', 'file_name', 'num_vox',
                                'num_pts']
                # output_columns=['feed_dict_list']
            )

    # from visualize import visualize_pcd
    # for fd_tuple in dataflow['train'].create_tuple_iterator(output_numpy=True):
    #     feed_dict = dataset['train'].collate_fn(*fd_tuple)
    #     lidar = feed_dict['lidar']
    #     num_vox = [80000, 80000]
    #     targets = ms.ops.cast(feed_dict['targets'].F, ms.int64)
    #     cur = 0
    #     for n in num_vox:
    #         pts = lidar.F[cur:cur+n, :3]
    #         label = targets[cur:cur+n]
    #         visualize_pcd(xyz=pts, target=label)
    #         cur += n

    net = builder.make_model()
    # net = SPVCNN_MS()
    def forward_fn(x, y):
        output = net(x)
        ce_ops = CrossEntropyLossWithIgnored(sparse=True, reduction='mean', ignore_index=255)
        ce = ce_ops(output, y)
        return ce

    for fd_tuple in dataflow['train'].create_tuple_iterator(output_numpy=True):
        feed_dict = dataset['train'].collate_fn(*fd_tuple)
        lidar = feed_dict['lidar']
        targets = feed_dict['targets'].F.astype(ms.int64)
        grad_fn = ms.value_and_grad(forward_fn, None, weights=net.trainable_params())
        loss, grad = grad_fn(lidar, targets)
        print(loss)

# if __name__ == '__main__':
#     x1 = Tensor(np.random.rand(92292, 4), dtype=ms.float32)
#     c1 = Tensor(np.random.randint(0, 1e5, size=(92292, 4)), dtype=ms.int32)
#     y = Tensor(np.random.randint(0, 19, size=(92292,)), dtype=ms.int64)
#     print(f"x1.shape:{x1.shape}, x1.dtype:{x1.dtype}")
#     print(f"c1.shape:{c1.shape}, c1.dtype:{c1.dtype}")
#
#     spvnas_test = SPVCNN_MS()
#     input = SparseTensor(x1, c1)
#     grad_fn = ms.value_and_grad(forward_fn, None, weights=spvnas_test.trainable_params())
#     loss, grad = grad_fn(input, y)
#     print(grad)

if __name__ == '__main__':
    main()





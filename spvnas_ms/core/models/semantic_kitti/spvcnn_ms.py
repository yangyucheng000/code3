import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Constant
from mindspore import ops
import torchsparse
from torchsparse import nn as spnn
from torchsparse import PointTensor, SparseTensor

from core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point

__all__ = ['SPVCNN_MS']

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

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

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

        self.classifier = nn.SequentialCell(nn.Dense(in_channels=cs[8], out_channels=kwargs['num_classes'], weight_init=None, bias_init=None))

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
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        exit()
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

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
#         if 'pres' in kwargs and 'vres' in kwargs:
#             self.pres = kwargs['pres']
#             self.vres = kwargs['vres']
#
#         self.net = nn.SequentialCell(
#             spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
#             spnn.BatchNorm(cs[0]),
#             spnn.ReLU())
#
#         self.classifier = nn.SequentialCell([nn.Dense(cs[0], kwargs['num_classes'])])
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
#
#         # print(f"net.conv3d: {z0.F}")
#         # print(f"net.conv3d.shape: {z0.F.shape}")
#         # print(f"conv3d success")
#
#         out = self.classifier(z0.F)
#         return out
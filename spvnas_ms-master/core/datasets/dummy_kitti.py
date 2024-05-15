import os
import os.path

import numpy as np
import mindspore as ms
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn, sparse_collate
from torchsparse.utils.quantize import sparse_quantize
from core.utils.config import configs

__all__ = ['DummyKITTI']

label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


class DummyKITTI(dict):

    def __init__(self, voxel_size, num_points, **kwargs):
        super().__init__({
            'train':
                DummyKITTIInternal(voxel_size,
                                   num_points,
                                   split='train'),
            'test':
                DummyKITTIInternal(
                    voxel_size,
                    num_points,
                    split='val')
        })


class DummyKITTIInternal:

    def __init__(self,
                 voxel_size,
                 num_points,
                 split):
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.seqs = []
        self.angle = 0.0

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return 100

    def __getitem__(self, index):
        xy = np.random.randn(80000, 2).astype(np.float32) * 75.0
        z = np.random.rand(80000, 1).astype(np.float32) * 5.0
        i = np.random.rand(80000, 1).astype(np.float32) * 255.0
        block_ = np.concatenate([xy, z, i], axis=-1)
        block = np.zeros_like(block_)

        if 'train' in self.split:
            # theta = np.random.uniform(0, 2 * np.pi)
            theta = 2.1847802132874214
            # print(f"dataloader.theta: {theta}")
            # scale_factor = np.random.uniform(0.95, 1.05)
            scale_factor = 1.0062435768866034
            # print(f"scale_factor: {scale_factor}")
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        labels_ = np.random.randint(0, 19, (80000,)).astype(np.int64)

        feat_ = block

        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)

        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]

        return (pc, feat, labels, pc_, labels_, inverse_map, "111111")


    @staticmethod
    def collate_fn(pc, feat, labels, pc_, labels_, inverse_map, file_name, num_vox, num_pts):

        batch = []
        batch_size = len(pc)
        file_name_list = str(file_name).split('\n')
        # print('batch size: ', batch_size)
        for i in range(batch_size):
            nv = num_vox[i]
            n = num_pts[i]
            input_dict = {
                'lidar': SparseTensor(feat[i], pc[i]),
                'targets': SparseTensor(labels[i], pc[i]),
                'targets_mapped': SparseTensor(labels_[i], pc_[i]),
                'inverse_map': SparseTensor(inverse_map[i], pc_[i]),
                'file_name': file_name_list[i]
            }
            batch.append(input_dict)
        return sparse_collate_fn(batch)

    @staticmethod
    def per_batch_map(pc, feat, labels, pc_, labels_, inverse_map, file_name, batchinfo):

        def _pad(data: list, pad_value=-1):
            # print("len(data):", len(data))
            # print('before pad:')
            max_size = data[0].shape[0]
            for d in data:
                # print('d.shape:', d.shape)
                if d.shape[0] > max_size:
                    max_size = d.shape[0]
            # print('max_size:', max_size)
            ds = list(data[0].shape)
            ds[0] = max_size
            padded_data = [np.full(shape=ds, fill_value=pad_value, dtype=data[0].dtype) for _ in data]
            for i, d in enumerate(data):
                padded_data[i][:d.shape[0]] = d
            return padded_data

        num_vox = [d.shape[0] for d in pc]
        num_pts = [d.shape[0] for d in pc_]

        return (_pad(pc), _pad(feat), _pad(labels), _pad(pc_), _pad(labels_), _pad(inverse_map), file_name, num_vox, num_pts)
# -*-coding:utf-8-*-
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import pickle
import random
import time
import argparse
import os
import glob
import numpy as np
import mindspore.dataset as ds
from os.path import join, exists
from helper_ply import read_ply
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP
from os import makedirs


class S3DISDatasetGenerator:
    def __init__(
        self,
        labeled_point,
        gt_save_path,
        data_type="ply",
        val_area="Area_5",
    ):
        self.path = "/home/ubuntu/hdd1/yhj/RPSC/Dataset/S3DIS"
        self.all_files = glob.glob(join(self.path, "original_ply", "*.ply"))
        self.data_type = data_type
        self.gt_path = gt_save_path

        self.label_to_names = {
            0: "ceiling",
            1: "floor",
            2: "wall",
            3: "beam",
            4: "column",
            5: "window",
            6: "door",
            7: "table",
            8: "chair",
            9: "sofa",
            10: "bookcase",
            11: "board",
            12: "clutter",
            13: "unlabel",
        }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([13])

        cfg.ignored_label_inds = [
            self.label_to_idx[ign_label] for ign_label in self.ignored_labels
        ]

        # Initiate containers
        self.input_trees = {"training": [], "validation": []}
        self.input_colors = {"training": [], "validation": []}
        self.input_labels = {"training": [], "validation": []}
        self.input_names = {"training": [], "validation": []}
        self.s_indx = {"training": [], "validation": []}
        self.val_proj = []
        self.val_labels = []
        self.val_area = val_area

        self.load_data(cfg.sub_grid_size, labeled_point)
        print("Size of training : ", len(self.input_colors["training"]))
        print("Size of validation : ", len(self.input_colors["validation"]))

    def load_data(self, sub_grid_size, labeled_point):
        tree_path = join(self.path, "input_{:.3f}".format(sub_grid_size))
        # Saved path of labeled points and their corresponding labels
        gt_path = self.gt_path
        makedirs(gt_path) if not exists(gt_path) else None

        for _, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split("/")[-1][:-4]
            if self.val_area in cloud_name:
                cloud_split = "validation"
            else:
                cloud_split = "training"

            # Name of the input files
            kd_tree_file = join(tree_path, "{:s}_KDTree.pkl".format(cloud_name))
            sub_ply_file = join(tree_path, "{:s}.ply".format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data["red"], data["green"], data["blue"])).T
            sub_labels = data["class"]

            all_select_label_indx = []
            if cloud_split == "training":
                all_select_label_indx = []
                for i in range(self.num_classes):
                    ind_class = np.where(sub_labels == i)[0]
                    num_classs = len(ind_class)
                    if num_classs > 0:
                        if "%" in labeled_point:
                            r = float(labeled_point[:-1]) / 100 * 14.6
                            num_selected = max(int(num_classs * r), 1)
                        else:
                            num_selected = int(labeled_point)
                        label_indx = list(range(num_classs))
                        random.shuffle(label_indx)
                        noselect_labels_indx = label_indx[num_selected:]
                        select_labels_indx = label_indx[:num_selected]
                        ind_class_noselect = ind_class[noselect_labels_indx]
                        ind_class_select = ind_class[select_labels_indx]
                        all_select_label_indx.append(ind_class_select[0])
                        sub_labels[ind_class_noselect] = 13

            # Read pkl with search tree
            with open(kd_tree_file, "rb") as f:
                search_tree = pickle.load(f)

            # The points information is in tree.data
            self.input_trees[cloud_split].append(search_tree)
            self.input_colors[cloud_split].append(sub_colors)
            self.input_labels[cloud_split].append(sub_labels)
            self.input_names[cloud_split].append(cloud_name)
            if cloud_split == "training":
                self.s_indx[cloud_split] += [all_select_label_indx]

            size = sub_colors.shape[0] * 4 * 10
            print(
                "{:s} {:.1f} MB loaded in {:.1f}s".format(
                    kd_tree_file.split("/")[-1], size * 1e-6, time.time() - t0
                )
            )

            ascii_name = join(gt_path, cloud_name)
            np.save(ascii_name, sub_labels)
            print(ascii_name + " has saved")
        print("\nPreparing reprojected indices for testing")

        # Get validation and test reprojected indices
        for _, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split("/")[-1][:-4]

            # Validation projection and labels
            if self.val_area in cloud_name:
                proj_file = join(tree_path, "{:s}_proj.pkl".format(cloud_name))
                with open(proj_file, "rb") as f:
                    proj_idx, labels = pickle.load(f)

                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print("{:s} done in {:.1f}s".format(cloud_name, time.time() - t0))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds
        return self.size


class ActiveLearningSampler(ds.Sampler):

    def __init__(self, dataset, batch_size=6, split="training"):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.possibility = {}
        self.min_possibility = {}

        if split == "training":
            self.n_samples = cfg.train_steps
        else:
            self.n_samples = cfg.val_steps

        # Random initialisation for weights
        self.possibility[split] = []
        self.min_possibility[split] = []
        for _, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        # not equal to the actual size of the dataset, but enable nice progress bars
        return self.n_samples * self.batch_size

    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.

        for _ in range(self.n_samples * self.batch_size):  # num_per_epoch
            # t0 = time.time()

            # Generator loop
            # Choose a random cloud
            cloud_idx = int(np.argmin(self.min_possibility[self.split]))

            # choose the point with the minimum of possibility as query point
            point_ind = np.argmin(self.possibility[self.split][cloud_idx])

            # Get points from tree structure
            points = np.array(
                self.dataset.input_trees[self.split][cloud_idx].data, copy=False
            )

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add noise to the center point
            noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
            pick_point = center_point + noise.astype(center_point.dtype)

            if len(points) < cfg.num_points:
                queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(
                    pick_point, k=len(points)
                )[1][0]
            else:
                queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(
                    pick_point, k=cfg.num_points
                )[1][0]

            if self.split == "training":
                s_indx = self.dataset.s_indx[self.split][cloud_idx]  # training only
                # Shuffle index
                queried_idx = np.concatenate([np.array(s_indx), queried_idx], 0)[
                    : cfg.num_points
                ]  # training only

            queried_idx = DP.shuffle_idx(queried_idx)
            # Collect points and colors
            queried_pc_xyz = points[queried_idx]
            queried_pc_xyz = queried_pc_xyz - pick_point
            queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][
                queried_idx
            ]
            queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][
                queried_idx
            ]

            dists = np.sum(
                np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1
            )
            delta = np.square(1 - dists / np.max(dists))
            self.possibility[self.split][cloud_idx][queried_idx] += delta
            self.min_possibility[self.split][cloud_idx] = float(
                np.min(self.possibility[self.split][cloud_idx])
            )

            if len(points) < cfg.num_points:
                queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = (
                    DP.data_aug(
                        queried_pc_xyz,
                        queried_pc_colors,
                        queried_pc_labels,
                        queried_idx,
                        cfg.num_points,
                    )
                )

            queried_pc_xyz = queried_pc_xyz.astype(np.float32)
            queried_pc_colors = queried_pc_colors.astype(np.float32)
            queried_pc_labels = queried_pc_labels.astype(np.int32)
            queried_idx = queried_idx.astype(np.int32)
            cloud_idx = np.array([cloud_idx], dtype=np.int32)

            yield queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cloud_idx


def ms_map(
    batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, batchInfo
):
    """
    xyz =>  [B,N,3]
    features =>  [B,N,d]
    labels =>  [B,N,]
    pc_idx =>  [B,N,]
    cloud_idx =>  [B,]
    """
    batch_xyz = np.array(batch_xyz)
    batch_features = np.array(batch_features)
    batch_features = np.concatenate((batch_xyz, batch_features), axis=-1)
    input_points = []  # [num_layers, B, N, 3]
    input_neighbors = []  # [num_layers, B, N, 16]
    input_pools = []  # [num_layers, B, N, 16]
    input_up_samples = []  # [num_layers, B, N, 1]

    for i in range(cfg.num_layers):
        neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n).astype(np.int32)
        sub_points = batch_xyz[:, : batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
        pool_i = neighbour_idx[:, : batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
        up_i = DP.knn_search(sub_points, batch_xyz, 1).astype(np.int32)
        input_points.append(batch_xyz)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)
        batch_xyz = sub_points

    # b_f:[B, N, 3+d]
    # due to the constraints of the mapping function, only the list elements can be passed back sequentially
    return (
        batch_features,
        batch_labels,
        batch_pc_idx,
        batch_cloud_idx,
        input_points[0],
        input_points[1],
        input_points[2],
        input_points[3],
        input_points[4],
        input_neighbors[0],
        input_neighbors[1],
        input_neighbors[2],
        input_neighbors[3],
        input_neighbors[4],
        input_pools[0],
        input_pools[1],
        input_pools[2],
        input_pools[3],
        input_pools[4],
        input_up_samples[0],
        input_up_samples[1],
        input_up_samples[2],
        input_up_samples[3],
        input_up_samples[4],
    )


def dataloader(dataset, **kwargs):
    val_sampler = ActiveLearningSampler(
        dataset, batch_size=cfg.val_batch_size, split="validation"
    )
    train_sampler = ActiveLearningSampler(
        dataset, batch_size=cfg.batch_size, split="training"
    )
    return (
        ds.GeneratorDataset(
            train_sampler,
            column_names=["xyz", "colors", "labels", "q_idx", "c_idx"],
            **kwargs
        ),
        ds.GeneratorDataset(
            val_sampler,
            column_names=["xyz", "colors", "labels", "q_idx", "c_idx"],
            **kwargs
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="the number of GPUs to use [default: 0]"
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="options: train, test, vis"
    )
    parser.add_argument(
        "--test_area",
        type=int,
        default=5,
        help="Which area to use for test, option: 1-6 [default: 5]",
    )
    parser.add_argument("--labeled_point", type=str, default="1", help="1, 10 or 100")
    parser.add_argument(
        "--model_name", type=str, default="RandLANet_S3DIS_pretrain.py", help=""
    )
    parser.add_argument("--log_dir", type=str, default="exx", help="")
    parser.add_argument("--knn", type=int, default=16, help="k_nn")
    parser.add_argument("--gt_save_path", type=str, default="./S3DIS_gt_2", help="")
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    Mode = FLAGS.mode

    test_area = FLAGS.test_area
    dataset = S3DISDatasetGenerator(FLAGS.labeled_point, FLAGS.gt_save_path)
    train, test = dataloader(dataset)
    train = train.batch(
        batch_size=cfg.batch_size,
        per_batch_map=ms_map,
        input_columns=["xyz", "colors", "labels", "q_idx", "c_idx"],
        output_columns=[
            "features",
            "labels",
            "input_inds",
            "cloud_inds",
            "p0",
            "p1",
            "p2",
            "p3",
            "p4",
            "n0",
            "n1",
            "n2",
            "n3",
            "n4",
            "pl0",
            "pl1",
            "pl2",
            "pl3",
            "pl4",
            "u0",
            "u1",
            "u2",
            "u3",
            "u4",
        ],
        drop_remainder=True,
    )
    i = 1
    for data in train.create_dict_iterator():
        print(data)
        break

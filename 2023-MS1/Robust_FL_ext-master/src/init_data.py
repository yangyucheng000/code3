# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
Filename: init_data.py
"""
import numpy as np
from src.cifar import CIFAR10, CIFAR100


class Cifar10FL:
    """
    Cifar10 for FL
    """

    def __init__(
            self,
            root,
            dataidxs=None,
            train=True,
            transform=None,
            target_transform=None,
            download=False,
            noise_type=None,
            noise_rate=0.1,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.data, self.target = self.construct()

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    def construct(self):
        """
        Construct Participant Dataset
        """
        cifar_dataobj = CIFAR10(
            self.root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
            self.noise_type,
            self.noise_rate,
        )
        if self.train:
            data = cifar_dataobj.train_data
            target = np.array(cifar_dataobj.train_noisy_labels)
        else:
            data = cifar_dataobj.test_data
            target = np.array(cifar_dataobj.test_labels)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target


class Cifar100FL:
    """
    Cifar100 for FL
    """

    def __init__(
            self,
            root,
            dataidxs=None,
            train=True,
            transform=None,
            target_transform=None,
            download=False,
            noise_type=None,
            noise_rate=0,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.data, self.target = self.construct()

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    def construct(self):
        """
        Construct Participant Dataset
        """
        cifar_dataobj = CIFAR100(
            self.root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
            self.noise_type,
            self.noise_rate,
        )
        if self.train:
            data = cifar_dataobj.train_data
            target = np.array(cifar_dataobj.train_noisy_labels)
        else:
            data = cifar_dataobj.test_data
            target = np.array(cifar_dataobj.test_labels)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

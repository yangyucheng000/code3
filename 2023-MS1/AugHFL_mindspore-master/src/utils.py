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
'''utils.py'''
import sys
import logging
import os
import random

import numpy as np
import mindspore.dataset as data
import mindspore.ops as ops
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.dataset.vision import py_transforms, Border

from src.init_data import Cifar10FL, Cifar100FL, CIFAR_C
from src.efficientnet import efficientnet
from src.resnet import resnet10, resnet12
from src.shufflenet import shufflenet
from dataaug import AugMixDataset, AugMixPublicDataset

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
PROJECT_PATH = r"./"


class CIFARDATA:
    """
    CIFAR Data
    """

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


class DOWNLOADERDATA:
    """
    Downloader Data
    """

    def __init__(self, train_dl, test_dl, train_ds, test_ds):
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.train_ds = train_ds
        self.test_ds = test_ds


def init_logs(log_level=logging.INFO, log_path=PROJECT_PATH + "Logs/", sub_name=None):
    """
    init_logs
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)  # Log等级总开关
    log_path = log_path
    mkdirs(log_path)
    filename = os.path.basename(sys.argv[0][0:-3])
    if sub_name is None:
        log_name = log_path + filename + ".log"
    else:
        log_name = log_path + filename + "_" + sub_name + ".log"
    logfile = log_name
    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    )
    fh.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(console)
    return logger


def mkdirs(dirpath):
    """
    mkdirs
    """
    if os.path.exists(dirpath) is False:
        os.makedirs(dirpath)


def load_cifar10_data(datadir, noise_type=None, noise_rate=0):
    """
    load cifar10 data
    """
    transform = Compose([py_transforms.ToTensor()])
    cifar10_train_ds = Cifar10FL(
        datadir,
        train=True,
        download=True,
        transform=transform,
        noise_type=noise_type,
        noise_rate=noise_rate,
    )
    cifar10_test_ds = Cifar10FL(
        datadir, train=False, download=True, transform=transform
    )
    x_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    x_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return CIFARDATA(x_train, y_train, x_test, y_test)


def load_cifar100_data(datadir, noise_type=None, noise_rate=0):
    """
    load cifar100 data
    """
    transform = Compose([py_transforms.ToTensor()])
    cifar100_train_ds = Cifar100FL(
        datadir,
        train=True,
        download=True,
        transform=transform,
        noise_type=noise_type,
        noise_rate=noise_rate,
    )
    cifar100_test_ds = Cifar100FL(
        datadir, train=False, download=True, transform=transform
    )
    x_train, y_train = cifar100_train_ds.data, cifar100_test_ds.target
    x_test, y_test = cifar100_train_ds.data, cifar100_test_ds.target
    return CIFARDATA(x_train, y_train, x_test, y_test)


def generate_public_data_indexs(dataset, datadir, size, noise_type=None, noise_rate=0):
    """
    generate public data indexes
    """
    if dataset == "cifar100":
        temp_data = load_cifar100_data(
            datadir, noise_type=noise_type, noise_rate=noise_rate
        )
        y_train = temp_data.y_train
    if dataset == "cifar10":
        temp_data = load_cifar10_data(
            datadir, noise_type=noise_type, noise_rate=noise_rate
        )
        y_train = temp_data.y_train
    n_train = y_train.shape[0]
    idxs = np.random.permutation(n_train)
    idxs = idxs[0:size]
    return idxs


def get_dataloader(
        dataset,
        datadir,
        train_bs,
        test_bs,
        dataidxs=None,
        noise_level=0,
        corruption_type=None,
        corruption_rate=0,
        test_corruption_type=None,
        test_corruption_rate=0
):
    """
    get dataloader
    """
    if dataset == "cifar10":
        dl_obj = Cifar10FL
        normalize = py_transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        transform_train = Compose(
            [
                py_transforms.ToPIL(),
                py_transforms.Pad(
                    padding=(4, 4, 4, 4), padding_mode=Border.REFLECT
                ),
                py_transforms.RandomColorAdjust(brightness=noise_level),
                py_transforms.RandomCrop(32),
                py_transforms.RandomHorizontalFlip(),
                py_transforms.ToTensor(),
                normalize,
            ]
        )
        train_ds = dl_obj(datadir,dataidxs=dataidxs,train=True,download=True)
    if dataset == "cifar100":
        dl_obj = CIFAR_C
        normalize = py_transforms.Normalize(
            mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
        )
        transform_train = Compose(
            [
                py_transforms.ToPIL(),
                py_transforms.RandomCrop(32, padding=4),
                py_transforms.RandomHorizontalFlip(),
                py_transforms.RandomRotation(15),
                py_transforms.ToTensor(),
                normalize,
            ]
        )
        train_ds = dl_obj(datadir, dataidxs=dataidxs, c_type='random_noise', transform=transform_train, corrupt_rate=1)
        train_ds = AugMixPublicDataset(train_ds, preprocess, corrupt='augmix')

    if dataset == 'cifar10c':
        dl_obj = CIFAR_C
        #Augmix transform
        transform_train = Compose([
            py_transforms.ToTensor(),
            (lambda x: ops.pad(ops.stop_gradient(x.unsqueeze(0)),
                               (4, 4, 4, 4), mode='reflect').data.squeeze()),
            py_transforms.ToPIL(),
            py_transforms.RandomColorAdjust(brightness=noise_level),
            py_transforms.RandomCrop(32),
            py_transforms.RandomHorizontalFlip()
        ])
        preprocess = Compose(
            [py_transforms.ToTensor(),
             py_transforms.Normalize([0.5] * 3, [0.5] * 3)])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, c_type=corruption_type, transform=transform_train, corrupt_rate=corruption_rate)
        train_ds = AugMixDataset(train_ds, preprocess, no_jsd=False)
    
    # For the test dataset:
    if test_dataset == 'clean':
        test_datadir = '../Dataset/cifar_10'
        normalize_test = py_transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_test = Compose([
            py_transforms.ToTensor(),
            normalize_test
        ])
        test_ds = Cifar10FL(test_datadir, train=False, transform=transform_test, download=False)
    if test_dataset == 'corrupt':
        test_datadir = '../Dataset/cifar_10/CIFAR-10-C_test'
        normalize_test = py_transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1944, 0.2010])
        transform_test = Compose([
            py_transforms.ToTensor(),
            normalize_test
        ])
        test_ds = CIFAR_C(test_datadir, c_type=test_corruption_type, transform=transform_test, corrupt_rate=test_corruption_rate)


    train_dl = data.GeneratorDataset(
        source=train_ds, column_names=["img", "target"], shuffle=True
    )
    train_dl = train_dl.map(transform_train)
    train_dl = train_dl.batch(batch_size=train_bs, drop_remainder=True)

    test_dl = data.GeneratorDataset(
        source=test_ds, column_names=["img", "target"], shuffle=False
    )
    test_dl = test_dl.map(transform_test)
    test_dl = test_dl.batch(batch_size=test_bs)

    return DOWNLOADERDATA(train_dl, test_dl, train_ds, test_ds)


def init_nets(n_parties, nets_name_list):
    """
    init nets
    """
    nets_list = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net_name = nets_name_list[net_i]
        if net_name == "ResNet10":
            net = resnet10()
        elif net_name == "ResNet12":
            net = resnet12()
        elif net_name == "ShuffleNet":
            net = shufflenet()
        elif net_name == "EfficientNet":
            net = efficientnet()
        nets_list[net_i] = net
    return nets_list

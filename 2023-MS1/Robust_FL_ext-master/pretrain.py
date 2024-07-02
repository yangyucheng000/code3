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
"""pretrain.py"""

import random
import yaml

import mindspore
from mindspore import nn
from mindspore import Model
import numpy as np
from numpy.core.fromnumeric import mean
from mindvision.engine.callback import LossMonitor

from src.utils import (
    init_logs,
    get_dataloader,
    init_nets,
    mkdirs,
)
from loss import SCELoss

with open("./config.yaml", "r", encoding="utf-8") as f:
    fdata = f.read()
    cfg = yaml.safe_load(fdata)

SEED = 0
n_participants = cfg["n_participants"]
train_batch_size = cfg["pretrain"]["train_batch_size"]
test_batch_size = cfg["pretrain"]["test_batch_size"]
pretrain_epoch = cfg["pretrain"]["pretrain_epoch"]
private_data_len = cfg["pretrain"]["private_data_len"]
pariticpant_params = cfg["pretrain"]["pariticpant_params"]

"""Noise Setting"""
noise_type = cfg["noise_type"]
noise_rate = cfg["noise_rate"]
"""Heterogeneous Model Setting"""
nets_name_list = ["ResNet10", "ResNet12", "ShuffleNet", "EfficientNet"]
DATASET_NAME = "cifar10"
DATASET_DIR = "./src/datasets"
dataset_classes = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def pretrain_network(
        epoch, net, data_loader, loss_function, optimizer_name, learning_rate
):
    """

    Args:
        epoch:
        net:
        data_loader:
        loss_function:
        optimizer_name:
        learning_rate:

    Returns:

    """
    if loss_function == "CE":
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    if loss_function == "SCE":
        criterion = SCELoss(alpha=0.4, beta=1, num_classes=10)

    if optimizer_name == "Adam":
        optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    if optimizer_name == "SGD":
        optimizer = nn.SGD(
            net.trainable_params(),
            learning_rate=learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )
    model = Model(net, loss_fn=criterion, optimizer=optimizer, metrics={"acc"})
    model.train(epoch, data_loader, callbacks=[LossMonitor(0.001, 1)])
    return net


def evaluate_network(net, dataloader, loss_function):
    """

    Args:
        net:
        dataloader:
        loss_function:

    Returns:

    """
    if loss_function == "CE":
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    if loss_function == "SCE":
        criterion = SCELoss(alpha=0.4, beta=1, num_classes=10)
    model = Model(net, loss_fn=criterion, metrics={"acc"})
    acc = model.eval(dataloader)

    print(f"Test Accuracy of the model on the test images: {100 * acc['acc']} %")
    return 100 * acc["acc"]


if __name__ == "__main__":
    mkdirs(
        "./Model_Storage/"
        + "pretrain/"
        + pariticpant_params["loss_funnction"]
        + "/"
        + str(noise_type)
        + str(noise_rate)
    )
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    random.seed(SEED)
    np.random.seed(SEED)

    logger.info("Load Participants' Data and Model")
    net_dataidx_map = {}
    for index in range(n_participants):
        idxes = np.random.permutation(50000)
        idxes = idxes[0:private_data_len]
        net_dataidx_map[index] = idxes
    logger.info(net_dataidx_map)
    net_list = init_nets(n_parties=n_participants, nets_name_list=nets_name_list)

    logger.info("Pretrain Participants Models")
    for index in range(n_participants):
        data = get_dataloader(
            dataset=DATASET_NAME,
            datadir=DATASET_DIR,
            train_bs=train_batch_size,
            test_bs=test_batch_size,
            dataidxs=net_dataidx_map.get(index),
            noise_type=noise_type,
            noise_rate=noise_rate,
        )
        train_dl_local = data.train_dl
        train_ds_local = data.train_ds
        network = net_list[index]
        netname = nets_name_list[index]
        logger.info(
            "Pretrain the %sth Participant Model with N_training: %s",
            index,
            len(train_ds_local),
        )
        network = pretrain_network(
            epoch=pretrain_epoch,
            net=network,
            data_loader=train_dl_local,
            loss_function=pariticpant_params["loss_funnction"],
            optimizer_name=pariticpant_params["optimizer_name"],
            learning_rate=pariticpant_params["learning_rate"],
        )
        logger.info("Save the %sth Participant Model", index)
        mindspore.save_checkpoint(
            network,
            "./Model_Storage/"
            + "pretrain/"
            + pariticpant_params["loss_funnction"]
            + "/"
            + str(noise_type)
            + str(noise_rate)
            + "/"
            + netname
            + "_"
            + str(index)
            + ".ckpt",
        )

    logger.info("Evaluate Models")
    test_accuracy_list = []
    for index in range(n_participants):
        data = get_dataloader(
            dataset=DATASET_NAME,
            datadir=DATASET_DIR,
            train_bs=train_batch_size,
            test_bs=test_batch_size,
            dataidxs=net_dataidx_map.get(index),
        )
        test_dl = data.test_dl
        network = net_list[index]
        netname = nets_name_list[index]

        param_dict = mindspore.load_checkpoint(
            "./Model_Storage/"
            + "pretrain/"
            + pariticpant_params["loss_funnction"]
            + "/"
            + str(noise_type)
            + str(noise_rate)
            + "/"
            + netname
            + "_"
            + str(index)
            + ".ckpt"
        )
        mindspore.load_param_into_net(network, param_dict)
        output = evaluate_network(net=network, dataloader=test_dl, loss_function="SCE")
        test_accuracy_list.append(output)
    print(
        "The average Accuracy of models on the test images:"
        + str(mean(test_accuracy_list))
    )

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
"""aughfl.py"""

import random
from cmath import exp
from statistics import mean

import yaml
import numpy as np
import mindspore
from mindspore import ops, nn
from mindspore.nn import WithLossCell

from src.utils import (
    init_logs,
    get_dataloader,
    init_nets,
    generate_public_data_indexs,
    mkdirs,
)
from loss import SCELoss, KLDivLoss

with open("./config.yaml", "r", encoding="utf-8") as f:
    fdata = f.read()
    cfg = yaml.safe_load(fdata)

SEED = 0
n_participants = cfg["n_participants"]
train_batch_size = cfg["aughfl"]["train_batch_size"]
test_batch_size = cfg["aughfl"]["test_batch_size"]
communication_epoch = cfg["aughfl"]["communication_epoch"]
pariticpant_params = cfg["aughfl"]["pariticpant_params"]

corruption_type = cfg["corruption_type"]
corruption_rate = cfg["corruption_rate"]
"""Heterogeneous Model Setting"""
private_nets_name_list = ["ResNet10", "ResNet12", "ShuffleNet", "EfficientNet"]
if corruption_rate == 0:
    PRIVATE_DATASET_NAME = 'cifar10'
    PRIVATE_DATA_DIR = './src/datasets'
else:
    PRIVATE_DATASET_NAME = 'cifar10c'
    PRIVATE_DATA_DIR = './src/dataset/CIFAR-10-C_train'
private_data_len = cfg["aughfl"]["private_data_len"]
private_dataset_classes = [
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
PRIVATE_OUTPUT_CHANNEL = len(private_dataset_classes)
PUBLIC_DATASET_NAME = "cifar100"
PUBLIC_DATASET_DIR = "./src/datasets"
public_dataset_length = cfg["aughfl"]["public_dataset_length"]


def evaluate_network(net, dataloader, log, loss_function):
    """

    Args:
        net:
        dataloader:
        log:
        loss_function:

    Returns:

    """
    if loss_function == "CE":
        loss_f = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    if loss_function == "SCE":
        loss_f = SCELoss(alpha=0.4, beta=1, num_classes=10)

    num_batches = dataloader.get_dataset_size()
    net.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data_eval, label in dataloader.create_tuple_iterator():
        pred = net(data_eval)
        total += len(data_eval)
        test_loss += loss_f(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    log.info(f"Test: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return 100 * correct


def update_model_via_private_data(
        net, epoch, private_dataloader, loss_function, optimizer_method, learning_rate, log
):
    """

    Args:
        net:
        epoch:
        private_dataloader:
        loss_function:
        optimizer_method:
        learning_rate:
        log:

    Returns:

    """
    if loss_function == "CE":
        loss_f = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    if loss_function == "SCE":
        loss_f = SCELoss(alpha=0.4, beta=1, num_classes=10)

    if optimizer_method == "Adam":
        optim = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    if optimizer_method == "SGD":
        optim = nn.SGD(
            net.trainable_params(),
            learning_rate=learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )
    t_net = nn.TrainOneStepCell(nn.WithLossCell(net, loss_f), optim)
    t_net.set_train()

    participant_local_loss_batch_list = []
    for epoch_i in range(epoch):
        for di in private_dataloader.create_dict_iterator():
            images = di["img"]
            labels = di["target"]
            #Augmix+JSD
            images_all = ops.cat(images, 0).cuda()
            logits_all,_ = t_net(x=images_all)
            logits_clean, logits_aug1, logits_aug2 = ops.split(
                logits_all, images[0].size(0))
            # Cross-entropy is only computed on clean images
            loss = t_net(logits_clean, labels.long())
            # loss = criterion(logits_clean, labels)
            p_clean, p_aug1, p_aug2 = ops.softmax(
                logits_clean, dim=1), ops.softmax(
                logits_aug1, dim=1), ops.softmax(
                logits_aug2, dim=1)
            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = ops.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (ops.kl_div(p_mixture, p_clean, reduction='batchmean') +
                          ops.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                          ops.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            participant_local_loss_batch_list.append(loss.asnumpy().item())
        log.info(f"Private Train Epoch: [{epoch_i} / {epoch}], " f"loss: {loss}")
    return net, participant_local_loss_batch_list


if __name__ == "__main__":
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    random.seed(SEED)
    np.random.seed(SEED)

    logger.info("Initialize Participants' Data idxs and Model")
    net_dataidx_map = {}
    for index in range(n_participants):
        idxes = np.random.permutation(50000)
        idxes = idxes[0:private_data_len]
        net_dataidx_map[index] = idxes
    logger.info(net_dataidx_map)

    net_list = init_nets(
        n_parties=n_participants, nets_name_list=private_nets_name_list
    )
    logger.info("Load Participants' Models")

    for i in range(n_participants):
        network = net_list[i]
        netname = private_nets_name_list[i]
        param_dict = mindspore.load_checkpoint(
            "./Model_Storage/"
            + "pretrain/"
            + pariticpant_params["loss_funnction"]
            + "/"
            + str(corruption_type)
            + str(corruption_rate)
            + "/"
            + netname
            + "_"
            + str(i)
            + ".ckpt"
        )
        mindspore.load_param_into_net(network, param_dict)

    logger.info("Initialize Public Data Parameters")
    public_data_indexs = generate_public_data_indexs(
        dataset=PUBLIC_DATASET_NAME,
        datadir=PUBLIC_DATASET_DIR,
        size=public_dataset_length
    )
    public_data = get_dataloader(
        dataset=PUBLIC_DATASET_NAME,
        datadir=PUBLIC_DATASET_DIR,
        train_bs=train_batch_size,
        test_bs=test_batch_size,
        dataidxs=public_data_indexs,
        corruption_type=corruption_type,
        corruption_rate=corruption_rate,
    )
    public_train_dl = public_data.train_dl
    public_train_ds = public_data.train_ds
    col_loss_list = []
    local_loss_list = []
    acc_list = []

    for epoch_index in range(communication_epoch):
        pariticpant_params["learning_rate"] = pariticpant_params["learning_rate"] * 0.96
        logger.info("The %sth Communication Epoch", epoch_index)

        logger.info("Evaluate Models")
        acc_epoch_list = []
        for participant_index in range(n_participants):
            netname = private_nets_name_list[participant_index]
            PRIVATE_DATASET_DIR = PRIVATE_DATA_DIR
            data = get_dataloader(
                dataset=PRIVATE_DATASET_NAME,
                datadir=PRIVATE_DATASET_DIR,
                train_bs=train_batch_size,
                test_bs=test_batch_size,
                dataidxs=None,
                corruption_type=corruption_type,
                corruption_rate=corruption_rate,
            )
            test_dl = data.test_dl
            network = net_list[participant_index]
            accuracy = evaluate_network(
                net=network, dataloader=test_dl, log=logger, loss_function="CE"
            )
            acc_epoch_list.append(accuracy)
        acc_list.append(acc_epoch_list)
        accuracy_avg = sum(acc_epoch_list) / n_participants

        for d in public_train_dl.create_dict_iterator():
            linear_output_list = []
            linear_output_target_list = []
            kl_loss_batch_list = []
            participant_kl_list = []
            """
            Calculate Linear Output
            """
            for participant_index in range(n_participants):
                network = net_list[participant_index]
                network.set_train()
                images = d["img"][participant_index]
                images_all = ops.cat(images, 0).cuda()
                logits_all = network(x=images_all)
                logits_clean, logits_aug1, logits_aug2 = ops.split(logits_all, images[0].size(0))
                p_clean, p_aug1, p_aug2 = ops.softmax(
                    logits_clean, dim=1), ops.softmax(
                    logits_aug1, dim=1), ops.softmax(
                    logits_aug2, dim=1)
                plog_clean = p_clean.log()
                linear_output_target_list.append(p_clean.clone().detach())
                linear_output_list.append(plog_clean)
                participant_kl = 1 / (ops.kl_div(p_aug1.log(), p_clean, reduction='batchmean') + ops.kl_div(p_aug2.log(), p_clean, reduction='batchmean'))
                participant_kl_list.append(participant_kl.clone().detach())
            participant_weight_list = []
            kl_sum = sum(participant_kl_list)
            for participant_index in range(n_participants):
                participant_weight_list.append(participant_kl_list[participant_index] / kl_sum)
            #HFL
            for participant_index in range(n_participants):
                network = net_list[participant_index]
                criterion = KLDivLoss()
                optimizer = nn.Adam(
                    network.trainable_params(),
                    learning_rate=pariticpant_params["learning_rate"],
                )
                loss = mindspore.Tensor(0)
                train_net = nn.TrainOneStepCell(
                    WithLossCell(network, criterion), optimizer
                )
                train_net.set_train()
                for i in range(n_participants):
                    if i != participant_index:
                        weight_index = participant_weight_list[i]
                        loss_batch_sample = criterion(
                            weight_index * linear_output_list[participant_index],
                            weight_index * linear_output_target_list[i],
                        )
                        loss = loss + loss_batch_sample

        for participant_index in range(n_participants):
            network = net_list[participant_index]
            network.set_train()
            private_dataidx = net_dataidx_map.get(participant_index)
            data = get_dataloader(
                dataset=PRIVATE_DATASET_NAME,
                datadir=PRIVATE_DATA_DIR,
                train_bs=train_batch_size,
                test_bs=test_batch_size,
                dataidxs=private_dataidx,
                corruption_type=corruption_type,
                corruption_rate=corruption_rate,
            )
            train_dl_local = data.train_dl
            train_ds_local = data.train_ds
            private_epoch = max(int(len(train_ds_local) / len(public_train_ds)), 1)

            network, private_loss_batch_list = update_model_via_private_data(
                net=network,
                epoch=private_epoch,
                private_dataloader=train_dl_local,
                loss_function=pariticpant_params["loss_funnction"],
                optimizer_method=pariticpant_params["optimizer_name"],
                learning_rate=pariticpant_params["learning_rate"],
                log=logger,
            )

        if epoch_index % 5 == 4 or epoch_index == communication_epoch - 1:
            mkdirs(
                "./Model_Storage/"
                + "AugHFL/"
                + pariticpant_params["loss_funnction"]
                + "/"
                + str(corruption_type)
                + str(corruption_rate)
            )
            logger.info("Save Models")
            for participant_index in range(n_participants):
                netname = private_nets_name_list[participant_index]
                network = net_list[participant_index]
                mindspore.save_checkpoint(
                    network,
                    "./Model_Storage/"
                    + "AugHFL/"
                    + pariticpant_params["loss_funnction"]
                    + "/"
                    + str(corruption_type)
                    + str(corruption_rate)
                    + "/"
                    + netname
                    + "_"
                    + str(participant_index)
                    + ".ckpt",
                )
    acc_epoch_list = []
    logger.info("Final Evaluate Models")
    for participant_index in range(n_participants):
        data = get_dataloader(
            dataset=PRIVATE_DATASET_NAME,
            datadir=PRIVATE_DATA_DIR,
            train_bs=train_batch_size,
            test_bs=test_batch_size,
            dataidxs=None,
            corruption_type=corruption_type,
            corruption_rate=corruption_rate,
        )
        test_dl = data.test_dl
        network = net_list[participant_index]
        accuracy = evaluate_network(
            net=network, dataloader=test_dl, log=logger, loss_function="CE"
        )
        acc_epoch_list.append(accuracy)
    acc_list.append(acc_epoch_list)
    accuracy_avg = sum(acc_epoch_list) / n_participants
    print(acc_list)
    print(accuracy_avg)

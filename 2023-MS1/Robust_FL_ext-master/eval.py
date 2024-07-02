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
'''eval.py'''

import yaml
import mindspore
from mindspore import nn

from src.utils import init_logs, get_dataloader, init_nets
from loss import SCELoss

with open("./config.yaml", "r", encoding="utf-8") as f:
    fdata = f.read()
    cfg = yaml.safe_load(fdata)

train_batch_size = cfg["rhfl"]["train_batch_size"]
test_batch_size = cfg["rhfl"]["test_batch_size"]
n_participants = cfg["n_participants"]
pariticpant_params = cfg["rhfl"]["pariticpant_params"]

noise_type = cfg["noise_type"]
noise_rate = cfg["noise_rate"]
"""Heterogeneous Model Setting"""
private_nets_name_list = ["ResNet10", "ResNet12", "ShuffleNet", "EfficientNet"]

PRIVATE_DATASET_NAME = "cifar10"
PRIVATE_DATA_DIR = "./src/datasets"


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
    for temp_data, label in dataloader.create_tuple_iterator():
        pred = net(temp_data)
        total += len(temp_data)
        test_loss += loss_f(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    log.info(f"Test: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return 100 * correct


if __name__ == "__main__":
    logger = init_logs()

    net_list = init_nets(
        n_parties=n_participants, nets_name_list=private_nets_name_list
    )
    logger.info("Load Models")

    for i in range(n_participants):
        network = net_list[i]
        netname = private_nets_name_list[i]
        param_dict = mindspore.load_checkpoint(
            "./Model_Storage/"
            + "RHFL/"
            + pariticpant_params["loss_funnction"]
            + "/"
            + str(noise_type)
            + str(noise_rate)
            + "/"
            + netname
            + "_"
            + str(i)
            + ".ckpt"
        )
        mindspore.load_param_into_net(network, param_dict)

    acc_list = []
    acc_epoch_list = []

    logger.info("Final Evaluate Models")
    for participant_index in range(n_participants):
        data = get_dataloader(
            dataset=PRIVATE_DATASET_NAME,
            datadir=PRIVATE_DATA_DIR,
            train_bs=train_batch_size,
            test_bs=test_batch_size,
            dataidxs=None,
            noise_type=noise_type,
            noise_rate=noise_rate,
        )
        test_dl = data.test_dl
        network = net_list[participant_index]
        accuracy = evaluate_network(
            net=network, dataloader=test_dl, log=logger, loss_function="SCE"
        )
        acc_epoch_list.append(accuracy)
    acc_list.append(acc_epoch_list)
    accuracy_avg = sum(acc_epoch_list) / n_participants
    print(acc_list)
    print(accuracy_avg)

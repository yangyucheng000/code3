# RHFL+ 描述

RHFL+（鲁棒异构联邦学习）是一种联邦学习框架，用于解决具有噪声和异构客户端的鲁棒联邦学习问题：

1. 在异构联邦学习中调整logits输出分布实现异构模型通信。
2. 设计抗噪声的鲁棒本地学习策略来对抗本地噪声。
3. 设计抗噪声反馈的协作学习策略对抗外部噪声。

# 数据集

使用的数据集：[CIFAR-10](<https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>)、[CIFAR-100](<https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz>)

## CIFAR-10

- 数据集大小：178 MB，由6万张32\*32的彩色图片组成，一共有10个类别。每个类别6000张图片
    - 训练集：148 MB，共50000组图像
    - 测试集：29.6 MB，共10000组图像

- 数据格式：用cPickle生成的python pickled对象
    - 注：数据将在src/init_data.py中加载和处理。

## CIFAR-100

- 数据集大小：178 MB，由6万张32\*32的彩色图片组成，一共有100个类别。每个类别600张图片
    - 训练集：148 MB，共50000组图像
    - 测试集：29.6 MB，共10000组图像

- 数据格式同上

- 数据目录树：官网下载数据后，解压压缩包，训练和评估所需的数据目录结构如下：

```shell
├─cifar-10-batches-py
│  ├─batches.meta
│  ├─data_batch_1
│  ├─data_batch_2
│  ├─data_batch_3
│  ├─data_batch_4
│  ├─data_batch_5
│  ├─readme.html
│  └─test_batch
│
└─cifar-100-python
   ├─file.txt~
   ├─meta
   ├─test
   └─train
```

# 环境要求

- 硬件（Ascend）
    - 使用Ascend处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Finstall)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Ftutorials%2Fzh-CN%2Fmaster%2Findex.html)
    - [MindSpore Python API](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Fdocs%2Fzh-CN%2Fmaster%2Findex.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估。

```shell
# 初始化数据集
python init.py
# 预训练本地模型
python pretrain.py
# RHFL
python rhfl.py
# 评估
python eval.py
```

# 脚本说明

## 脚本和样例代码

```shell
├── RHFL
    ├── src                 数据集和模型相关工具
        ├── cifar.py            数据集基类
        ├── init_data.py        数据集初始化类
        ├── utils.py            相关工具
        ├── efficientnet.py     客户机网络模型
        ├── resnet.py           客户机网络模型
        └── shufflenet.py       客户机网络模型
    ├── pretrain.py         预训练脚本
    ├── rhfl.py             联邦训练脚本
    ├── init.py             数据集初始化脚本
    ├── eval.py             评估脚本
    ├── export.py           导出模型脚本
    ├── loss.py             损失函数
    └── config.yaml         训练参数
```

## 脚本参数

在config.yaml中可以同时配置训练参数和评估参数。

可以直接查看config.yaml内的配置说明，说明如下

```yaml
n_participants:  客户机数目
noise_type:  噪声类别['pairflip','symmetric',None]
noise_rate:  噪声率

pretrain: 预训练相关参数
    train_batch_size: 训练batch大小
    test_batch_size: 测试batch大小
    pretrain_epoch: 预训练每个客户机迭代轮数
    private_data_len: 私有数据集长度
    pariticpant_params:
        loss_funnction: 损失函数
        optimizer_name: 优化器
        learning_rate: 初始学习率


rhfl: 联邦训练相关参数
    train_batch_size: 训练batch大小
    test_batch_size: 测试batch大小
    communication_epoch: 联邦训练轮数
    private_data_len: 私有数据集长度
    public_dataset_length: 公共数据集长度
    pariticpant_params:
        loss_funnction: 损失函数
        optimizer_name: 优化器
        learning_rate: 初始学习率
```

# 训练过程

## 训练

在昇腾上运行下面的命令进行训练

```shell
# 预训练本地模型
python pretrain.py
# RHFL
python rhfl_ext.py
```

训练过程日志

```log
2023-03-22 15:01:08,237 - rhfl_ext.py[line:148] - INFO: Random Seed and Server Config
2023-03-22 15:01:08,240 - rhfl_ext.py[line:152] - INFO: Initialize Participants' Data idxs and Model
2023-03-22 15:01:08,248 - rhfl_ext.py[line:158] - INFO: {0: array([11841, 19602, 45519, ..., 47278, 37020,  2217]), 1: array([28921, 11971, 15919, ..., 46749, 41193, 22418]), 2: array([36011, 11643, 45353, ..., 42473, 40043, 26550]), 3: array([ 8777,  6584, 42707, ..., 25134, 39808, 32076])}
2023-03-22 15:01:10,502 - rhfl_ext.py[line:161] - INFO: Load Participants' Models
2023-03-22 15:01:11,111 - rhfl_ext.py[line:172] - INFO: Initialize Public Data Parameters
```

训练结果日志
训练checkpoint将被保存在`Model_Storage/RHFL/Loss函数名（SCE,CE）/噪声类型+噪声率`中，你可以从如下的log文件中获取结果，log文件被保存在/Logs中。

```log
2023-03-22 15:01:21,158 - rhfl_ext.py[line:356] - INFO: Final Evaluate Models
2023-03-22 15:02:01,005 - rhfl_ext.py[line:105] - INFO: Test: Accuracy: 77.1%, Avg loss: 2.888967
2023-03-22 15:02:16,866 - rhfl_ext.py[line:105] - INFO: Test: Accuracy: 77.2%, Avg loss: 2.835910
2023-03-22 15:03:01,027 - rhfl_ext.py[line:105] - INFO: Test: Accuracy: 68.5%, Avg loss: 3.767210
2023-03-22 15:03:33,415 - rhfl_ext.py[line:105] - INFO: Test: Accuracy: 76.8%, Avg loss: 2.708377
```

# 评估

## 评估过程

在昇腾上运行下面的命令进行评估

```shell
python eval.py
```

## 评估结果

上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```log
2023-03-25 14:53:21,628 - eval.py[line:78] - INFO: Load Participants' Models
2023-03-25 14:53:22,198 - eval.py[line:93] - INFO: Final Evaluate Models
2023-03-25 14:54:01,454 - eval.py[line:69] - INFO: Test: Accuracy: 77.1%, Avg loss: 2.888967
2023-03-25 14:54:18,234 - eval.py[line:69] - INFO: Test: Accuracy: 77.2%, Avg loss: 2.835910
2023-03-25 14:55:04,642 - eval.py[line:69] - INFO: Test: Accuracy: 68.5%, Avg loss: 3.767210
2023-03-25 14:55:40,091 - eval.py[line:69] - INFO: Test: Accuracy: 76.8%, Avg loss: 2.708377
```

# 导出

## 导出过程

```shell
# 导出MINDIR和AIR
python export.py
```

## 导出结果

```log
2023-03-25 15:01:52,038 - export.py[line:45] - INFO: Load Participants' Models
2023-03-25 15:01:54,673 - export.py[line:64] - INFO: ResNet10_0model exported successfully
2023-03-25 15:01:56,260 - export.py[line:64] - INFO: ResNet12_1model exported successfully
2023-03-25 15:02:04,916 - export.py[line:64] - INFO: ShuffleNet_2model exported successfully
2023-03-25 15:02:11,037 - export.py[line:64] - INFO: EfficientNet_3model exported successfully
2023-03-25 15:02:11,040 - export.py[line:66] - INFO: Models exported successfully

```

# 推理

## 推理过程

```shell
python eval.py
```

## 推理结果

上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```log
2023-03-25 14:53:21,628 - eval.py[line:78] - INFO: Load Participants' Models
2023-03-25 14:53:22,198 - eval.py[line:93] - INFO: Final Evaluate Models
2023-03-25 14:54:01,454 - eval.py[line:69] - INFO: Test: Accuracy: 77.1%, Avg loss: 2.888967
2023-03-25 14:54:18,234 - eval.py[line:69] - INFO: Test: Accuracy: 77.2%, Avg loss: 2.835910
2023-03-25 14:55:04,642 - eval.py[line:69] - INFO: Test: Accuracy: 68.5%, Avg loss: 3.767210
2023-03-25 14:55:40,091 - eval.py[line:69] - INFO: Test: Accuracy: 76.8%, Avg loss: 2.708377
```

# 性能

## 训练性能

| Parameters                 | Ascend 910                                                   |
| -------------------------- | ------------------------------------------------------------ |
| Model Version              | ResNet10 ResNet12 ShuffleNet  Mobilenetv2                    |
| Resource                   | NPU: 1*Ascend 910, CPU: 24, Video Memory: 32GB, Memory: 256GB |
| uploaded Date              | 03/25/2023                                                   |
| MindSpore Version          | 1.8.1                                                        |
| Dataset                    | CIFAR-10                                                     |
| Training Parameters        | train_batch_size: 512    test_batch_size: 512    communication_epoch: 40 |
| Optimizer                  | Adam                                                         |
| Loss Function              | SCE                                                          |
| Loss                       | 3.050116                                                     |
| Speed                      | 1700 ms/step                                                 |
| Total time                 | 7hours                                                       |
| Checkpoint for Fine tuning | 56.9M (four .ckpt files)                                     |

## 推理性能

| Parameters          | Ascend                                                       |
| ------------------- | ------------------------------------------------------------ |
| Model Version       | ResNet10 ResNet12 ShuffleNet  Mobilenetv2                    |
| Resource            | NPU: 1*Ascend 910, CPU: 24, Video Memory: 32GB, Memory: 256GB |
| Uploaded Date       | 03/25/2023                                                   |
| MindSpore Version   | 1.8.1                                                        |
| Dataset             | CIFAR-10                                                     |
| batch_size          | 512                                                          |
| Accuracy            | 74.9075%                                                       |
| Model for inference | 56.9M (four .air files)                                              |

# 随机情况说明

在pretrain.py, rhfl_ext.py，eval.py中，我们设置了random.seed(0)和np.random.seed(0)种子。

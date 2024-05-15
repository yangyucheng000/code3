import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.train import Model
from mindspore.nn import Conv2d, LSTM, Dense
from mindspore.nn.loss import L1Loss
from mindspore.nn.metrics import MeanAbsoluteError
import mindspore.ops.operations as P
from sklearn.preprocessing import MinMaxScaler

# 设置设备
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 读取数据
data = np.genfromtxt('init.csv', delimiter=',', skip_header=1)
X = data[:, :10]  # 前10列是特征
y = data[:, 10]   # 第11列是目标

# 数据归一化预处理
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
y_normalized = scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

# 转换数据形状为适合CNN的形状
X_normalized = X_normalized.reshape(-1, 1, 5, 2)

# 构建CNN-LSTM模型
class CNN_LSTM_Model(nn.Cell):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        self.conv = Conv2d(1, 32, (3, 1), pad_mode='valid')
        self.relu = nn.ReLU()
        self.lstm = LSTM(input_size=32, hidden_size=64, num_layers=1, has_bias=True, batch_first=True)
        self.dense = Dense(64, 1)

    def construct(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.squeeze(2)  # 去掉单维度的卷积输出
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # 只取序列最后一个时间点的输出
        output = self.dense(output)
        return output

def main():
    # 准备数据
    X = Tensor(X_normalized.astype(np.float32))
    y = Tensor(y_normalized.astype(np.float32).reshape(-1, 1))

    # 定义模型、损失函数和优化器
    net = CNN_LSTM_Model()
    loss_fn = L1Loss()
    optimizer = nn.Adam(net.trainable_params(), learning_rate=0.001)

    # 训练模型
    model = Model(net, loss_fn, optimizer, metrics={"MAE": MeanAbsoluteError()})
    model.train(epochs=50, train_dataset=X, train_labels=y)

    # 预测
    predicted = net(X)

    # 反归一化
    predicted_denormalized = scaler.inverse_transform(predicted.asnumpy().reshape(-1, 1)).reshape(-1)
    print("Predicted prices (denormalized):", predicted_denormalized)

if __name__ == "__main__":
    main()

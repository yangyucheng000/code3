import shap
import numpy as np
import matplotlib.pyplot as plt
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.train import Model
from mindspore.nn import LSTM, Dense
from mindspore.nn.loss import L1Loss
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

# 构建LSTM模型
class LSTMModel(nn.Cell):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = LSTM(input_size=10, hidden_size=64, num_layers=1, has_bias=True, batch_first=True)
        self.dense = Dense(64, 1)

    def construct(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]  # 只取序列最后一个时间点的输出
        output = self.dense(output)
        return output
#实现对LSTM模型的可解释性分析
def main1():
    # 准备数据
    X = Tensor(X_normalized.astype(np.float32))
    y = Tensor(y_normalized.astype(np.float32).reshape(-1, 1))

    # 定义模型、损失函数和优化器
    net = LSTMModel()
    loss_fn = L1Loss()
    optimizer = nn.Adam(net.trainable_params(), learning_rate=0.001)

    # 训练模型
    model = Model(net, loss_fn, optimizer)
    model.train(epochs=50, train_dataset=X, train_labels=y)

    # 创建SHAP解释器
    explainer = shap.DeepExplainer(model, X)

    # 解释模型
    shap_values = explainer.shap_values(X)

    # 生成热力图
    shap.summary_plot(shap_values, X, feature_names=['Feature_{}'.format(i) for i in range(10)])
def main2():
    #导入CNN-LSTM模型#

    # 创建SHAP解释器
    explainer = shap.DeepExplainer(CNN-LSTM, X)

    # 解释模型
    shap_values = explainer.shap_values(X)

    # 生成热力图
    shap.summary_plot(shap_values, X, feature_names=['Feature_{}'.format(i) for i in range(10)])

#context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
# 加载模型
#class CNNLSTM(nn.Cell):
#    def __init__(self, input_size, hidden_size, num_layers=1):
#        super(CNNLSTM, self).__init__()
#        self.cnn = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, has_bias=True, batch_first=True)
#        self.fc = nn.Dense(hidden_size, 1)
#    def construct(self, x):
#        x = x.reshape((-1, 1, 5, 2))  # 调整输入数据的形状以适应CNN层
#        output = self.cnn(x)
#        output = output.reshape((-1, 5, 32))  # 调整输出数据的形状以适应LSTM层
#        output, (_, _) = self.lstm(output)
#        output = output[:, -1, :]
#        output = self.fc(output)
#        return output
#model = CNNLSTM(10, 64)
#load_checkpoint('cnn_lstm.ckpt', net=model)
#model.set_train(False)
# 加载数据
#data = np.genfromtxt('init.csv', delimiter=',', skip_header=1)
#x = data[:, :10]
#x = Tensor(x, mstype.float32)
# 创建SHAP解释器
#explainer = shap.DeepExplainer(model, x)
# 解释模型预测
#shap_values = explainer.shap_values(x)
# 打印特征重要性
#shap.summary_plot(shap_values, x, feature_names=['feature1', 'feature2', ...])
# 打印每个样本的特征重要性
#shap.force_plot(explainer.expected_value, shap_values[0], x[0], feature_names=['feature1', 'feature2', ...])

if __name__ == "__main__":
    #main1()
    #main2()
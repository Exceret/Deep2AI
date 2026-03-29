"""
Title: learning5.py
Content: 使用 FullConnectedLayer 实现MNIST手写数字识别

! Deprecated because `fc` cannot be imported. No relevant issues or info found in git repos.
! Neither in the internet.
"""

import numpy as np
from datetime import datetime
from typing import Literal, Optional
from learning4 import train_and_evaluate

def ts_print(
    message: str,
    symbol: Optional[Literal["info", "success", "warning", "error", "debug"]] = None,
    color: bool = True,
) -> None:
    """
    带时间戳的信息输出函数

    Args:
        message: 要输出的消息内容
        symbol: CLI 符号类型，可选值: 'info', 'success', 'warning', 'error', 'debug'，默认无符号
        color: 是否启用 ANSI 颜色输出（默认 True）
    """
    timestamp = datetime.now().strftime("[%Y/%m/%d %H:%M:%S]")

    symbols: dict = {
        "info": ("ℹ", "\033[36m" if color else ""),  # Cyan
        "success": ("✔", "\033[32m" if color else ""),  # Green
        "warning": ("⚠", "\033[33m" if color else ""),  # Yellow
        "error": ("✖", "\033[31m" if color else ""),  # Red
        "debug": ("◼", "\033[35m" if color else ""),  # Magenta
    }

    t_prefix: str = f"{timestamp} "
    if symbol in symbols:
        sym, col = symbols[symbol]
        symbol_prefix: str = f"{col}{sym} \033[0m" if color else f"{sym} "
    else:
        symbol_prefix: str = ""

    print(symbol_prefix + f"{t_prefix}{message}")


class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        """
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        """
        前向计算
        input_array: 输入向量，维度必须等于input_size
        """
        # 式2
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        """
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        """
        # 式8
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        """
        使用梯度下降算法更新权重
        """
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad


class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)

    # 神经网络类
    class Network(object):
        def __init__(self, layers):
            """
            构造函数
            """
            self.layers = []
            for i in range(len(layers) - 1):
                self.layers.append(
                    FullConnectedLayer(layers[i], layers[i + 1], SigmoidActivator())
                )

    def predict(self, sample):
        """
        使用神经网络实现预测
        sample: 输入样本
        """
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        """
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (
            label - self.layers[-1].output
        )
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)


def main() -> None:
    train_and_evaluate()
    ts_print("main() finished", "success")


if __name__ == "__main__":
    main()

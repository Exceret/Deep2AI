"""
Title: learning7.py
Content: 循环神经网络
"""

from typing import Callable
from functools import reduce
import numpy as np
from learning6 import ReluActivator, IdentityActivator, element_wise_op
from utils.ts_print import ts_print


class RecurrentLayer(object):
    def __init__(self, input_width, state_width, activator, learning_rate):
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        self.times = 0  # 当前时刻初始化为t0
        self.state_list = []  # 保存各个时刻的state
        self.state_list.append(np.zeros((state_width, 1)))  # 初始化s0
        self.U = np.random.uniform(-1e-4, 1e-4, (state_width, input_width))  # 初始化U
        self.W = np.random.uniform(-1e-4, 1e-4, (state_width, state_width))  # 初始化W
        self.gradient_list = []  # 保存各个时刻的权重梯度
        self.delta_list = []
        self.gradient = []

    def forward(self, input_array):
        """
        根据『式2』进行前向计算
        """
        self.times += 1
        state = np.dot(self.U, input_array) + np.dot(self.W, self.state_list[-1])
        element_wise_op(state, self.activator.forward)
        self.state_list.append(state)

    def backward(self, sensitivity_array, activator):
        """
        实现BPTT算法
        """
        self.calc_delta(sensitivity_array, activator)
        self.calc_gradient()

    def update(self):
        """
        按照梯度下降，更新权重
        """
        self.W -= self.learning_rate * self.gradient

    def calc_delta(self, sensitivity_array, activator):
        """_summary_

        Args:
            sensitivity_array (_type_): _description_
            activator (_type_): _description_
        """
        self.delta_list = []  # 用来保存各个时刻的误差项
        self.delta_list = [np.zeros((self.state_width, 1)) for _ in range(self.times)]
        self.delta_list.append(sensitivity_array)
        # 迭代计算每个时刻的误差项
        for k in range(self.times - 1, 0, -1):
            self.calc_delta_k(k, activator)

    def calc_delta_k(self, k, activator):
        """
        根据k+1时刻的delta计算k时刻的delta
        """
        state = self.state_list[k + 1].copy()
        element_wise_op(self.state_list[k + 1], activator.backward)
        self.delta_list[k] = np.dot(
            np.dot(self.delta_list[k + 1].T, self.W), np.diag(state[:, 0])
        ).T

    def calc_gradient(self):
        """_summary_"""
        self.gradient_list = []  # 保存各个时刻的权重梯度
        for t in range(self.times + 1):
            self.gradient_list.append(np.zeros((self.state_width, self.state_width)))
        for t in range(self.times, 0, -1):
            self.calc_gradient_t(t)
        # 实际的梯度是各个时刻梯度之和
        self.gradient = reduce(
            lambda a, b: a + b, self.gradient_list, self.gradient_list[0]
        )  # [0]被初始化为0且没有被修改过

    def calc_gradient_t(self, t):
        """
        计算每个时刻t权重的梯度
        """
        gradient = np.dot(self.delta_list[t], self.state_list[t - 1].T)
        self.gradient_list[t] = gradient

    def reset_state(self):
        """_summary_"""
        self.times = 0  # 当前时刻初始化为t0
        self.state_list = []  # 保存各个时刻的state
        self.state_list.append(np.zeros((self.state_width, 1)))  # 初始化s0


def data_set():
    """_summary_

    Returns:
        _type_: _description_
    """
    x = [np.array([[1], [2], [3]]), np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x, d


def gradient_check():
    """
    梯度检查
    """
    # 设计一个误差函数，取所有节点输出项之和
    # error_function = lambda o: o.sum()

    rl = RecurrentLayer(3, 2, IdentityActivator(), 1e-3)

    # 计算forward值
    x: list[np.ndarray,np.ndarray] = data_set()[0]
    # x = data_set()
    rl.forward(x[0])
    rl.forward(x[1])

    # 求取sensitivity map
    sensitivity_array = np.ones(rl.state_list[-1].shape, dtype=np.float64)
    # 计算梯度
    rl.backward(sensitivity_array, IdentityActivator())

    # 检查梯度
    epsilon = 10e-4
    for i in range(rl.W.shape[0]):
        for j in range(rl.W.shape[1]):
            rl.W[i, j] += epsilon
            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            err1 = rl.state_list[-1].sum()
            rl.W[i, j] -= 2 * epsilon
            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            err2 = rl.state_list[-1].sum()
            expect_grad = (err1 - err2) / (2 * epsilon)
            rl.W[i, j] += epsilon
            print(
                f"weights({i},{j}): expected - actural {expect_grad} - {rl.gradient[i, j]}"
            )


def test():
    """_summary_

    Returns:
        _type_: _description_
    """
    layer = RecurrentLayer(3, 2, ReluActivator(), 1e-3)
    x, d = data_set()
    layer.forward(x[0])
    layer.forward(x[1])
    layer.backward(d, ReluActivator())
    return layer


def main() -> None:
    """_summary_"""
    layer: RecurrentLayer = test()

    print(layer)

    gradient_check()

    ts_print("main() finished", "success")


if __name__ == "__main__":
    main()

"""
Title: learning6.py
Content: 卷积神经网络

! deprecated because of the lack of `activator`
"""

from utils.ts_print import ts_print
import numpy as np
from typing import List, Tuple, Callable, Union
# from activator import ReluActivator, IdentityActivator
# from activations import


def relu(x) -> np.array:
    return np.maximum(0, x)


# ============ 替换 activator 包的激活函数类 ============
class ReluActivator:
    """ReLU: f(x) = max(0, x), f'(x) = 1 if x > 0 else 0"""

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        # 注意：backward 接收的是原始输入 x，返回导数值
        # 在 x=0 处导数未定义，这里约定为 0（与 PyTorch 一致）
        return (x > 0).astype(np.float64)


class IdentityActivator:
    """Identity: f(x) = x, f'(x) = 1"""

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x, dtype=np.float64)


# ====================================================


def get_patch(
    input_array: np.ndarray,
    i: int,
    j: int,
    filter_width: int,
    filter_height: int,
    stride: int,
) -> np.ndarray:
    """
    从输入数组中获取本次卷积的区域，
    自动适配输入为2D和3D的情况
    """
    start_i: int = i * stride
    start_j: int = j * stride

    if input_array.ndim == 1:
        raise ValueError("input_array.ndim must be 2 or 3, but got 1")

    if input_array.ndim == 2:
        return input_array[
            start_i : start_i + filter_height, start_j : start_j + filter_width
        ]
    elif input_array.ndim == 3:
        return input_array[
            :, start_i : start_i + filter_height, start_j : start_j + filter_width
        ]


def get_max_index(array: np.ndarray) -> Tuple[int, int]:
    """
    获取一个2D区域的最大值所在的索引
    """
    max_i: int = 0
    max_j: int = 0
    max_value: float = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_i = i
                max_j = j
                max_value = array[i, j]
    return max_i, max_j


def conv(
    input_array: np.ndarray,
    kernel_array: np.ndarray,
    output_array: np.ndarray,
    stride: int,
    bias: float,
) -> np.ndarray:
    """
    计算卷积，自动适配输入为2D和3D的情况
    """

    # channel_type :int = input_array.ndim
    ouput_width: int = output_array.shape[1]
    ouput_height: int = output_array.shape[0]
    kernel_width: int = kernel_array.shape[-1]
    kernel_height: int = kernel_array.shape[-2]
    for i in range(ouput_height):
        for j in range(ouput_width):
            output_array[i][j] = (
                get_patch(
                    input_array=input_array,
                    i=i,
                    j=j,
                    filter_width=kernel_width,
                    filter_height=kernel_height,
                    stride=stride,
                )
                * kernel_array
            ).sum() + bias


def padding(input_array: np.ndarray, zp: int) -> np.ndarray:
    """
    为数组增加Zero padding，自动适配输入为2D和3D的情况
    """
    if zp == 0:
        return input_array

    if input_array.ndim == 3:
        input_width: int = input_array.shape[2]
        input_height: int = input_array.shape[1]
        input_depth: int = input_array.shape[0]
        # ts_print(f"zp = {zp} (suspicious int)", "warning")
        # zp: int = int(zp)
        padded_array: np.ndarray = np.zeros(
            shape=(input_depth, input_height + 2 * zp, input_width + 2 * zp)
        )
        padded_array[:, zp : zp + input_height, zp : zp + input_width] = (
            input_array
        )
        return padded_array

    if input_array.ndim == 2:
        input_width: int = input_array.shape[1]
        input_height: int = input_array.shape[0]
        padded_array: np.ndarray = np.zeros(
            shape=(input_height + 2 * zp, input_width + 2 * zp)
        )
        padded_array[zp : zp + input_height, zp : zp + input_width] = input_array
        return padded_array

    raise ValueError(
        "input_array.ndim must be 2 or 3, but got {}".format(input_array.ndim)
    )


def element_wise_op(array: np.ndarray, op: Callable[[np.ndarray], np.ndarray]) -> None:
    """
    # 对numpy数组进行element wise操作
    """
    # for i in np.nditer(array, op_flags=["readwrite"]):
    #     i[...] = op(i)
    """对数组进行向量化激活函数操作（替代逐元素循环）"""
    array[...] = op(array)  # 直接对整个数组操作，利用 NumPy 广播


class Filter(object):
    def __init__(self, width: int, height: int, depth: int) -> None:
        self.weights = np.random.uniform(
            low=-1e-4, high=1e-4, size=(depth, height, width)
        )
        self.bias: float = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad: float = 0

    def __repr__(self) -> str:
        return "filter weights:\n%s\nbias:\n%s" % (repr(self.weights), repr(self.bias))

    def get_weights(self) -> np.ndarray:
        return self.weights

    def get_bias(self) -> float:
        return self.bias

    def update(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


class ConvLayer(object):
    def __init__(
        self,
        input_width: int,
        input_height: int,
        channel_number: int,
        filter_width: int,
        filter_height: int,
        filter_number: int,
        zero_padding: int,
        stride: int,
        activator: Union["ReluActivator", "IdentityActivator"],
        learning_rate: float,
    ) -> None:
        self.input_width: int = input_width
        self.input_height: int = input_height
        self.channel_number: int = channel_number
        self.filter_width: int = filter_width
        self.filter_height: int = filter_height
        self.filter_number: int = filter_number
        self.zero_padding: int = zero_padding
        self.stride: int = stride
        self.output_width: float = ConvLayer.calculate_output_size(
            self.input_width, filter_width, zero_padding, stride
        )
        self.output_height: float = ConvLayer.calculate_output_size(
            self.input_height, filter_height, zero_padding, stride
        )
        self.output_array: np.ndarray = np.zeros(
            (self.filter_number, int(self.output_height), int(self.output_width))
        )
        self.filters: List[Filter] = []
        for i in range(filter_number):
            self.filters.append(
                Filter(filter_width, filter_height, self.channel_number)
            )
        self.activator = activator
        self.learning_rate: float = learning_rate

    def forward(self, input_array: np.ndarray) -> None:
        """
        计算卷积层的输出
        输出结果保存在self.output_array
        """
        self.input_array = input_array
        self.padded_input_array: np.ndarray = padding(input_array, self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(
                self.padded_input_array,
                filter.get_weights(),
                self.output_array[f],
                self.stride,
                filter.get_bias(),
            )
        element_wise_op(self.output_array, self.activator.forward)

    def backward(
        self,
        input_array: np.ndarray,
        sensitivity_array: np.ndarray,
        activator: Union["ReluActivator", "IdentityActivator"],
    ) -> None:
        """
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        """
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    def update(self) -> None:
        """
        按照梯度下降，更新权重
        """
        for filter in self.filters:
            filter.update(self.learning_rate)

    def bp_sensitivity_map(
        self,
        sensitivity_array: np.ndarray,
        activator: Union["ReluActivator", "IdentityActivator"],
    ) -> None:
        """
        计算传递到上一层的sensitivity map
        sensitivity_array: 本层的sensitivity map
        activator: 上一层的激活函数
        """
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        # full卷积，对sensitivitiy map进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差
        # 但这个残差不需要继续向上传递，因此就不计算了
        expanded_width = expanded_array.shape[2]
        zp: int = round((self.input_width + self.filter_width - 1 - expanded_width) / 2)
        padded_array = padding(expanded_array, zp)
        # 初始化delta_array，用于保存传递到上一层的
        # sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的
        # sensitivity map相当于所有的filter的
        # sensitivity map之和
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重翻转180度
            flipped_weights = np.array(
                list(map(lambda i: np.rot90(i, 2), filter.get_weights()))
            )
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_array: np.ndarray) -> None:
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            # 计算每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(
                    self.padded_input_array[d],
                    expanded_array[f],
                    filter.weights_grad[d],
                    1,
                    0,
                )
            # 计算偏置项的梯度
            filter.bias_grad = expanded_array[f].sum()

    def expand_sensitivity_map(self, sensitivity_array: np.ndarray) -> np.ndarray:
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        expanded_width = (
            self.input_width - self.filter_width + 2 * self.zero_padding + 1
        )
        expanded_height = (
            self.input_height - self.filter_height + 2 * self.zero_padding + 1
        )
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(int(self.output_height)):
            for j in range(int(self.output_width)):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self) -> np.ndarray:
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    @staticmethod
    def calculate_output_size(
        input_size: int, filter_size: int, zero_padding: int, stride: int
    ) -> float:
        return (input_size - filter_size + 2 * zero_padding) / stride + 1


class MaxPoolingLayer(object):
    def __init__(
        self,
        input_width: int,
        input_height: int,
        channel_number: int,
        filter_width: int,
        filter_height: int,
        stride: int,
    ) -> None:
        self.input_width: int = input_width
        self.input_height: int = input_height
        self.channel_number: int = channel_number
        self.filter_width: int = filter_width
        self.filter_height: int = filter_height
        self.stride: int = stride
        self.output_width: float = (input_width - filter_width) / self.stride + 1
        self.output_height: float = (input_height - filter_height) / self.stride + 1
        self.output_array: np.ndarray = np.zeros(
            (self.channel_number, int(self.output_height), int(self.output_width))
        )

    def forward(self, input_array: np.ndarray) -> None:
        for d in range(self.channel_number):
            for i in range(int(self.output_height)):
                for j in range(int(self.output_width)):
                    self.output_array[d, i, j] = get_patch(
                        input_array[d],
                        i,
                        j,
                        self.filter_width,
                        self.filter_height,
                        self.stride,
                    ).max()

    def backward(self, input_array: np.ndarray, sensitivity_array: np.ndarray) -> None:
        self.delta_array: np.ndarray = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(int(self.output_height)):
                for j in range(int(self.output_width)):
                    patch_array = get_patch(
                        input_array[d],
                        i,
                        j,
                        self.filter_width,
                        self.filter_height,
                        self.stride,
                    )
                    max_i, max_j = get_max_index(patch_array)
                    self.delta_array[
                        d, i * self.stride + max_i, j * self.stride + max_j
                    ] = sensitivity_array[d, i, j]


def init_test() -> Tuple[np.ndarray, np.ndarray, ConvLayer]:
    a = np.array(
        [
            [
                [0, 1, 1, 0, 2],
                [2, 2, 2, 2, 1],
                [1, 0, 0, 2, 0],
                [0, 1, 1, 0, 0],
                [1, 2, 0, 0, 2],
            ],
            [
                [1, 0, 2, 2, 0],
                [0, 0, 0, 2, 0],
                [1, 2, 1, 2, 1],
                [1, 0, 0, 0, 0],
                [1, 2, 1, 1, 1],
            ],
            [
                [2, 1, 2, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 2, 1, 0, 1],
                [0, 1, 2, 2, 2],
                [2, 1, 0, 0, 1],
            ],
        ]
    )
    b = np.array([[[0, 1, 1], [2, 2, 2], [1, 0, 0]], [[1, 0, 2], [0, 0, 0], [1, 2, 1]]])
    cl = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    cl.filters[0].weights = np.array(
        [
            [[-1, 1, 0], [0, 1, 0], [0, 1, 1]],
            [[-1, -1, 0], [0, 0, 0], [0, -1, 0]],
            [[0, 0, -1], [0, 1, 0], [1, -1, -1]],
        ],
        dtype=np.float64,
    )
    cl.filters[0].bias = 1
    cl.filters[1].weights = np.array(
        [
            [[1, 1, -1], [-1, -1, 1], [0, -1, 1]],
            [[0, 1, 0], [-1, 0, -1], [-1, 1, 0]],
            [[-1, 0, 0], [-1, 0, 1], [-1, 0, 0]],
        ],
        dtype=np.float64,
    )
    return a, b, cl


def test() -> None:
    a, b, cl = init_test()
    cl.forward(a)
    print(cl.output_array)


def test_bp() -> None:
    a, b, cl = init_test()
    cl.backward(a, b, IdentityActivator())
    cl.update()
    print(cl.filters[0])
    print(cl.filters[1])


def gradient_check() -> None:
    """
    梯度检查
    """

    # 设计一个误差函数，取所有节点输出项之和
    def error_function(o: np.ndarray) -> float:
        o.sum()

    # 计算forward值
    a, b, cl = init_test()
    cl.forward(a)

    # 求取sensitivity map
    sensitivity_array = np.ones(cl.output_array.shape, dtype=np.float64)
    # 计算梯度
    cl.backward(a, sensitivity_array, IdentityActivator())
    # 检查梯度
    epsilon = 10e-4
    for d in range(cl.filters[0].weights_grad.shape[0]):
        for i in range(cl.filters[0].weights_grad.shape[1]):
            for j in range(cl.filters[0].weights_grad.shape[2]):
                cl.filters[0].weights[d, i, j] += epsilon
                cl.forward(a)
                
                # ts_print(f"type of `cl.output_array` is {type(cl.output_array)}","warning")
                # print(cl.output_array)
                
                # err1 = error_function(cl.output_array)
                err1:float = cl.output_array.sum()
                
                # ts_print(f"type of `err1` is {type(err1)}","warning")
                # print(err1)
                
                cl.filters[0].weights[d, i, j] -= 2 * epsilon
                cl.forward(a)
                # err2 = error_function(cl.output_array)
                err2 = cl.output_array.sum()

                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].weights[d, i, j] += epsilon
                print(
                    "weights(%d,%d,%d): expected - actural %f - %f"
                    % (d, i, j, expect_grad, cl.filters[0].weights_grad[d, i, j])
                )


def init_pool_test() -> Tuple[np.ndarray, np.ndarray, MaxPoolingLayer]:
    a = np.array(
        [
            [[1, 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]],
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1], [3, 4, 5, 6]],
        ],
        dtype=np.float64,
    )

    b = np.array([[[1, 2], [2, 4]], [[3, 5], [8, 2]]], dtype=np.float64)

    mpl = MaxPoolingLayer(4, 4, 2, 2, 2, 2)

    return a, b, mpl


def test_pool() -> None:
    a, b, mpl = init_pool_test()
    mpl.forward(a)
    print("input array:\n%s\noutput array:\n%s" % (a, mpl.output_array))


def test_pool_bp() -> None:
    a, b, mpl = init_pool_test()
    mpl.backward(a, b)
    print(
        "input array:\n%s\nsensitivity array:\n%s\ndelta array:\n%s"
        % (a, b, mpl.delta_array)
    )


def main() -> None:

    # test()
    # test_bp()
    # gradient_check()
    # test_pool()
    test_pool_bp()
    ts_print("main() finished", "success")


if __name__ == "__main__":
    main()

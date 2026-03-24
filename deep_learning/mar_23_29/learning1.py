"""
Title: learning1.py
Content: 实现感知器
"""

from utils.ts_print import ts_print
from typing import Callable


class Perceptron(object):
    def __init__(
        self: "Perceptron", input_num: int, activator_fun: Callable[[float], float]
    ) -> str:
        self.activator: Callable = activator_fun
        self.weights = [0.0] * input_num
        self.bias: float = 0.0
        print("initial weight:{0}, bias:{1}".format(self.weights, self.bias))

    def __str__(self) -> None:
        """
        打印学习到的权重、偏置项
        """
        return "weights: {0}\tbias: {1}\n".format(self.weights, self.bias)

    def predict(self: "Perceptron", input_vec) -> int:
        """
        输入向量，输出感知器的计算结果
        """

        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用sum求和
        zipped = list(zip(input_vec, self.weights))
        sum_total = sum(list(map(lambda x_y: x_y[0] * x_y[1], zipped)))
        return self.activator(sum_total + self.bias)

    def train(self, input_vecs, labels, iteration, rate) -> None:
        """
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate) -> None:
        """
        一次迭代，把所有的训练数据过一遍
        """
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = list(zip(input_vecs, labels))
        # 对每个样本，按照感知器规则更新权重
        for input_vec, label in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate) -> None:
        """
        按照感知器规则更新权重
        """
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        print(
            f"input_vec:{input_vec}, output:{output:>2}, label:{label:>2}, rate:{rate:>2}"
        )
        delta = label - output
        self.weights = list(
            map(
                lambda x_w: rate * delta * x_w[0] + x_w[1], zip(input_vec, self.weights)
            )
        )
        # 更新bias
        self.bias += rate * delta
        print(f"weights:{self.weights}, bias:{self.bias:>2}, delta:{delta:>2}")


def f_active_function(x) -> int:
    """
    定义激活函数f
    """
    return 1 if x > 0 else 0


def get_training_dataset() -> [list[list[int]], list[int]]:
    """
    基于and真值表构建训练数据
    """
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1, 0, 1, 1]
    return input_vecs, labels


def train_and_perceptron() -> Perceptron:
    """
    使用and真值表训练感知器
    """
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p: Perceptron = Perceptron(2, f_active_function)
    # 训练，迭代10轮, 学习速率为0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    # 返回训练好的感知器
    return p


def main() -> None:

    and_perception: Perceptron = train_and_perceptron()

    print(f"&感知器: {and_perception}")
    print("input value{0}, predict:{1}".format([0, 0], and_perception.predict([0, 0])))
    print(and_perception.predict([1, 0]))
    print(and_perception.predict([0, 1]))
    print(and_perception.predict([1, 1]))

    ts_print(message="Main() finished", symbol="success")


if __name__ == "main":
    main()

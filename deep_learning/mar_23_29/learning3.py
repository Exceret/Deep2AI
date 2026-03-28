"""
Title: learning3.py
Content: 实现神经网络
"""


import random
from numpy import exp
from functools import reduce
from typing import List, Tuple
from utils.ts_print import ts_print


def sigmoid(inX: float) -> float:
    return 1.0 / (1 + exp(-inX))


class Node(object):
    def __init__(self, layer_index: int, node_index: int) -> None:
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream: List["Connection"] = []
        self.upstream: List["Connection"] = []
        self.output: float = 0
        self.delta: float = 0

    def set_output(self, output: float) -> None:
        self.output = output

    def append_downstream_connection(self, conn: "Connection") -> None:
        self.downstream.append(conn)

    def append_upstream_connection(self, conn: "Connection") -> None:
        self.upstream.append(conn)

    def calc_output(self) -> None:
        # 每个节点的输出算法，N元一次方程求和
        output = reduce(
            lambda ret, conn: ret + conn.upstream_node.output * conn.weight,
            self.upstream,
            0,
        )
        # 结果放入激活函数
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self) -> None:
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream,
            0.0,
        )
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label: float) -> None:
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self) -> str:
        node_str = "%u-%u: output: %f delta: %f" % (
            self.layer_index,
            self.node_index,
            self.output,
            self.delta,
        )
        downstream_str = reduce(
            lambda ret, conn: ret + "\n\t" + str(conn), self.downstream, ""
        )
        upstream_str = reduce(
            lambda ret, conn: ret + "\n\t" + str(conn), self.upstream, ""
        )
        return (
            node_str
            + "\n\tdownstream:"
            + downstream_str
            + "\n\tupstream:"
            + upstream_str
        )


class ConstNode(object):
    def __init__(self, layer_index: int, node_index: int) -> None:
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream: List["Connection"] = []
        self.output: float = 1

    def append_downstream_connection(self, conn: "Connection") -> None:
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self) -> None:
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream,
            0.0,
        )
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self) -> str:
        node_str = "%u-%u: output: 1" % (self.layer_index, self.node_index)
        downstream_str = reduce(
            lambda ret, conn: ret + "\n\t" + str(conn), self.downstream, ""
        )
        return node_str + "\n\tdownstream:" + downstream_str


class Connection(object):
    def __init__(self, upstream_node: Node, downstream_node: Node) -> None:
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight: float = random.uniform(-0.1, 0.1)
        self.gradient: float = 0.0

    def calc_gradient(self) -> None:
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self) -> float:
        return self.gradient

    def update_weight(self, rate: float) -> None:
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self) -> str:
        return "(%u-%u) -> (%u-%u): weight: %f, gradient: %f" % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight,
            self.gradient,
        )


class Layer(object):
    def __init__(self, layer_index: int, node_count: int) -> None:
        self.layer_index = layer_index
        self.nodes: List[Node] = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))

    def set_output(self, data: List[float]) -> None:
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self) -> None:
        for node in self.nodes:
            node.calc_output()

    def dump(self) -> None:
        for node in self.nodes:
            print(node)


class Network(object):
    def __init__(self, layers: List[int]) -> None:
        self.connections = Connections()
        self.layers: List[Layer] = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            if i != layer_count - 1:
                # 非输出层，额外添加一个偏置节点(ConstNode)
                self.layers.append(Layer(i, layers[i] + 1))
                node_count += layers[i] + 1
            else:
                # 输出层，不添加偏置节点
                self.layers.append(Layer(i, layers[i]))
                node_count += layers[i]
        for i in range(layer_count - 1):
            # 连接当前层和下一层的节点
            connections = []
            for upstream_node in self.layers[i].nodes:
                for downstream_node in self.layers[i + 1].nodes:
                    conn = Connection(upstream_node, downstream_node)
                    connections.append(conn)
                    upstream_node.append_downstream_connection(conn)
                    downstream_node.append_upstream_connection(conn)
            for conn in connections:
                self.connections.add_connection(conn)

    def train(
        self,
        labels: List[List[float]],
        data_set: List[List[float]],
        rate: float,
        epoch: int,
    ) -> None:
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(
        self, label: List[float], sample: List[float], rate: float
    ) -> None:
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label: List[float]) -> None:
        # 计算输出层delta
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        # 计算隐藏层delta（从后往前）
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate: float) -> None:
        # 从后往前逐层更新权重
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self) -> None:
        # 计算所有连接的梯度
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label: List[float], sample: List[float]) -> None:
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample: List[float]) -> List[float]:
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes))

    def dump(self) -> None:
        for layer in self.layers:
            layer.dump()

    def __str__(self) -> str:
        lines = []
        for i, layer in enumerate(self.layers):
            lines.append("Layer %d: %d nodes" % (i, len(layer.nodes)))
        return "\n".join(lines)


class Connections(object):
    def __init__(self) -> None:
        self.connections: List[Connection] = []

    def add_connection(self, connection: Connection) -> None:
        self.connections.append(connection)


class Normalizer(object):
    def __init__(self) -> None:
        self.mask: List[int] = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    def norm(self, number: int) -> List[float]:
        return list(map(lambda m: 0.9 if number & m else 0.1, self.mask))

    def denorm(self, vec: List[float]) -> int:
        binary: list[int] = list(map(lambda i: 1 if i > 0.5 else 0, vec))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)


def mean_square_error(vec1: List[float], vec2: List[float]) -> float:
    return 0.5 * reduce(
        lambda a, b: a + b,
        list(map(lambda v: (v[0] - v[1]) * (v[0] - v[1]), zip(vec1, vec2))),
    )


def gradient_check(
    network: Network, sample_feature: List[float], sample_label: List[float]
) -> None:
    """
    梯度检查
    network: 神经网络对象
    sample_feature: 样本的特征
    sample_label: 样本的标签
    """
    # 计算网络误差
    network_error = lambda vec1, vec2: (
        0.5
        * reduce(
            lambda a, b: a + b,
            list(map(lambda v: (v[0] - v[1]) * (v[0] - v[1]), zip(vec1, vec2))),
        )
    )

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()

        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)

        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)

        # 打印
        print(
            "expected gradient: \t%f\nactual gradient: \t%f"
            % (expected_gradient, actual_gradient)
        )


def train_data_set() -> Tuple[List[List[float]], List[List[float]]]:
    normalizer: Normalizer = Normalizer()
    data_set: list = []
    labels: list = []
    # 生成不重复的随机样本，自编码器任务：输入=输出
    numbers: list[int] = random.sample(range(256), 32)
    for n in numbers:
        vec = normalizer.norm(n)
        data_set.append(vec)
        labels.append(vec)
    return labels, data_set


def train(network: Network) -> None:
    assert isinstance(network, object)
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.3, 50)


def test(network: Network, data: int) -> None:
    normalizer: Normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print("\ttestdata(%u)\tpredict(%u)" % (data, normalizer.denorm(predict_data)))


def correct_ratio(network: Network) -> None:
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print("correct_ratio: %.2f%%" % (correct / 256 * 100))


def gradient_check_test() -> None:
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == "__main__":
    # gradient_check_test()
    # 设置神经网络初始化参数，初始化神经网络

    # net: Network = Network([6, 4, 2])
    net = Network([8, 3, 8])
    print(net)
    train(net)
    net.dump()
    
    gradient_check_test()
    
    # correct_ratio(net)
    ts_print("Main() finished", "success")

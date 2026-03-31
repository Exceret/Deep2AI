"""
Title: learning4.py
Content: 实现MNIST手写数字识别
"""

import struct

# from fc import *
from learning3 import Network
from datetime import datetime
from typing import Literal, Optional

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


# 数据加载器基类
class Loader(object):
    def __init__(self, path, count) -> None:
        """
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        """
        self.path = path
        self.count = count

    def get_file_content(self) -> bytes:
        """
        读取文件内容
        """
        f = open(self.path, "rb")
        content: bytes = f.read()
        f.close()
        return content

    def to_int(self, byte: bytes) -> int:
        """
        将unsigned byte字符转换为整数
        """
        # return struct.unpack("B", byte)[0]
        if isinstance(byte, int):
            return byte
        return struct.unpack("B", byte)[0]


# 图像数据加载器
class ImageLoader(Loader):
    def get_picture(self, content: bytes, index: int) -> list[list[int]]:
        """
        内部函数，从文件中获取图像
        """
        start: int = index * 28 * 28 + 16
        picture: list = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(self.to_int(content[start + i * 28 + j]))
        return picture

    def get_one_sample(self, picture) -> list[list[int]]:
        """
        内部函数，将图像转化为样本的输入向量
        """
        sample: list = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self) -> list[list[int]]:
        """
        加载数据文件，获得全部样本的输入向量
        """
        content: bytes = self.get_file_content()
        data_set: list = []
        for index in range(self.count):
            data_set.append(self.get_one_sample(self.get_picture(content, index)))
        return data_set


# 标签数据加载器
class LabelLoader(Loader):
    def load(self) -> list[list[int]]:
        """
        加载数据文件，获得全部样本的标签向量
        """
        content: bytes = self.get_file_content()
        labels: list = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label) -> list[int]:
        """
        内部函数，将一个值转换为10维标签向量
        """
        label_vec: list = []
        label_value: int = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec


def get_training_data_set():
    """
    获得训练数据集
    """
    image_loader = ImageLoader(r"data/MNIST/train-images-idx3-ubyte", 60000)
    label_loader = LabelLoader(r"data/MNIST/train-labels-idx1-ubyte", 60000)
    return image_loader.load(), label_loader.load()


def get_test_data_set():
    """
    获得测试数据集
    """
    image_loader = ImageLoader(r"data/MNIST/t10k-images-idx3-ubyte", 10000)
    label_loader = LabelLoader(r"data/MNIST/t10k-labels-idx1-ubyte", 10000)
    return image_loader.load(), label_loader.load()


def show(sample):
    str: str = ""
    for i in range(28):
        for j in range(28):
            if sample[i * 28 + j] != 0:
                str += "*"
            else:
                str += " "
        str += "\n"
    print(str)


def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)

    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)


def now():
    return datetime.now().strftime("%c")


def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()

    # 为了加快训练，只使用前一部分数据（可选）
    # train_data_set = train_data_set[:10000]
    # train_labels = train_labels[:10000]

    network = Network([784, 100, 10])
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.01, 1)
        ts_print(
            "%s epoch %d finished, loss %f"
            % (
                now(),
                epoch,
                network.loss(train_labels[-1], network.predict(train_data_set[-1])),
            ),
            "info",
        )
        if epoch % 2 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print("%s after epoch %d, error ratio is %f" % (now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                ts_print("Early stopping triggered!", "warning")
                break
            else:
                last_error_ratio = error_ratio

            # if epoch >= 20:
            #     ts_print("Max epoch reached!","warning")
            #     break


def main() -> None:
    train_and_evaluate()
    ts_print("main() finished", "success")


if __name__ == "__main__":
    main()

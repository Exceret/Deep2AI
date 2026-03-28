"""
Title: learning4.py
Content: 实现MNIST手写数字识别

! Deprecated because `fc` cannot be imported. No relevant issues or info found in git repos. 
! Neither in the internet. 
"""

from utils.ts_print import ts_print
import struct
# from fc import *
from learning3 import Network
from datetime import datetime


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
        label_value: list[int] = self.to_int(label)
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
    image_loader = ImageLoader(r"data/MNIST/train-images-idx3-ubyte", 10000)
    label_loader = LabelLoader(r"data/MNIST/train-labels-idx1-ubyte", 10000)
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

def transpose(data_tuple):
    """
    对数据元组中的二维列表进行转置
    输入：(images, labels)，其中 images 是 [N, 784], labels 是 [N,10]
    输出：(transposed_images, transposed_labels)，其中 images 是 [784, N], labels 是 [10, N]
    """
    images, labels = data_tuple
    
    # 使用 zip(*data) 实现二维列表转置，并转换回 list 类型
    # images: (60000, 784) -> (784, 60000)
    transposed_images = [list(x) for x in zip(*images)]
    
    # labels: (60000, 10) -> (10, 60000)
    transposed_labels = [list(x) for x in zip(*labels)]
    
    return transposed_images, transposed_labels


def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = transpose(get_training_data_set())
    test_data_set, test_labels = transpose(get_test_data_set())
    network = Network([784, 100, 10])
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.01, 1)
        print(
            "%s epoch %d finished, loss %f"
            % (
                now(),
                epoch,
                network.loss(train_labels[-1], network.predict(train_data_set[-1])),
            )
        )
        if epoch % 2 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print("%s after epoch %d, error ratio is %f" % (now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio


def main() -> None:
    train_and_evaluate()
    ts_print("main() finished", "success")


if __name__ == "__main__":
    main()

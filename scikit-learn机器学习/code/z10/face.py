import logging
import time
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from matplotlib import pyplot as plt

# 更改logging日志模块的默认行为
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# 读取数据
data_home = 'E:\Data\code\datasets'
logging.info("开始读取数据集")
# Load the Olivetti faces data-set from AT&T (classification).
faces = fetch_olivetti_faces(data_home=data_home)
logging.info("读取完成")

# 统计
X = faces.data  # 数据
y = faces.target  # 标签.这里给出的实际是索引号
targets = np.unique(y)  # 标签只留一个
target_names = np.array(["c%d" % t for t in targets])  # 用索引号给人物命名为"c索引号"
n_targets = target_names.shape[0]  # 类别数
n_samples, h, w = faces.images.shape  # 样本数,图像高,图像宽
print("样本数:{}\n标签种类数:{}".format(n_samples, n_targets))
print("图像尺寸:宽{}高{}\n数据集X的shape:{}".format(w, h, X.shape))


def plot_gallery(images, titles, h, w, n_row=2, n_col=5):
    """绘制图片阵列"""
    plt.figure(figsize=(2 * n_col, 2.2 * n_row), dpi=100)
    # 跳帧子图布局,这里hspace指子图之间高度h上的间隔
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.01)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap='gray')
        plt.title(titles[i])
        plt.axis('off')  # 不提供坐标轴


def look_face():
    """查看人脸图片阵列"""
    n_row = 2
    n_col = 6
    sample_images = None
    sample_titles = []
    # 对于每个可能的标签
    for i in range(len(targets)):
        # 选取出该标签的所有样本
        people_images = X[y == targets[i]]
        # 随机从中选取出一个样本,即对这个人随机选它的一张照片
        people_sample_index = np.random.randint(0, people_images.shape[0], 1)
        people_sample_image = people_images[people_sample_index, :]
        # (不是第一个放进来的人)
        if sample_images is not None:
            # 这时要用np.concatenate()做数组拼接,即将这个人(的特征数组)放在下一行
            sample_images = np.concatenate((sample_images, people_sample_image), axis=0)
        # (是第一个放进来的人)
        else:
            sample_images = people_sample_image
        # 将标签放入标签列表中
        sample_titles.append(target_names[i])
    # 调用绘制图片阵列的函数来绘制这些人脸图像的图片阵列
    plot_gallery(sample_images, sample_titles, h, w, n_row, n_col)
    plt.show()

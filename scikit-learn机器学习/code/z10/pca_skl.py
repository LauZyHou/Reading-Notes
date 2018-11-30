from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def init():
    global A
    # 原始矩阵
    A = np.array([
        [3, 2000],
        [2, 3000],
        [4, 5000],
        [5, 8000],
        [1, 2000]
    ])
    # 这里修改其中元素的类型为float64,否则将给MinMaxScaler()内部做转换会引起警告
    # By default(copy=True), astype always returns a newly allocated array.
    A = A.astype(np.float64, copy=False)


# 预处理->PCA管道
def std_PCA(**kwargs):
    scalar = MinMaxScaler()  # 用于数据预处理(归一化和缩放)
    pca = PCA(**kwargs)  # PCA本身不包含预处理
    pipline = Pipeline([('scalar', scalar), ('pca', pca)])
    return pipline


if __name__ == '__main__':
    init()
    # 使用(这里n_components指定降维后的维数k)
    pca = std_PCA(n_components=1)
    Z = pca.fit_transform(A)
    print(Z)

    # 数据还原
    A_approx = pca.inverse_transform(Z)
    print(A_approx)

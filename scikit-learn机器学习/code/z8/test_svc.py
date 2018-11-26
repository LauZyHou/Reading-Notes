from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import numpy as np


# 绘制分类超平面,其中X是有两个特征的,所以实际绘制出来就是在平面上的不同类别区域用不同颜色标记
# 这里h是采样步长,draw_sv指示是否绘制支持向量
def plot_hyperplane(clf, X, y,
                    h=0.02,
                    draw_sv=True,
                    title='hyperplan'):
    # 绘制的区间,x轴和y轴都从最小值-1到最大值+1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 使用np.meshgrid()扩充为两轴的所有可能取值的组合
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    # np.ravel()将采样点的xy坐标摊平,使用np.c_将其按列组合成[[x y][x y]...]的坐标点形式
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # 用绘制等高线图的方式来绘制不同类别为不同颜色(等高线图上同一高度为同一颜色)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    # 不同类(标签)展示在图上的的标记和颜色
    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    # 可能的类(标签)取值
    labels = np.unique(y)
    # 对于每种标签取值
    for label in labels:
        # 绘制相应的样本点,使用自己的标记和颜色
        plt.scatter(X[y == label][:, 0],
                    X[y == label][:, 1],
                    c=colors[label],
                    marker=markers[label])
    # 绘制支持向量
    if draw_sv:
        # 用该方式可以直接取出支持向量是哪些点
        sv = clf.support_vectors_
        # 绘制为白色'x',这样就会贴在之前的有色点上了
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')


# 生成聚类样本100个,特征数为2(默认n_features=2),类别数为2,标准差0.3,随机种子设为0
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.3)
# print(X,y)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

plt.figure(figsize=(12, 4), dpi=144)
plot_hyperplane(clf, X, y, h=0.01, title="最大分类超平面")
plt.show()

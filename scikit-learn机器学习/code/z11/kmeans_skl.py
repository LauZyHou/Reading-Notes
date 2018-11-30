from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def show_sample():
    """展示样本点"""
    plt.figure(figsize=(6, 4), dpi=100)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(X[:, 0], X[:, 1], s=20, marker='o')
    plt.show()


def fit_plot_kmeans_model(k, X):
    """使样本X聚k类,并绘制图像"""
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    # 这里score得到的是一个负数,其绝对值越大表示成本越高
    # sklearn中对该成本的计算为:样例到其所属的聚类中心点的距离总和(而不是平均值)
    plt.title("k={},成本={}".format(k, kmeans.score(X)))
    # 聚类得到的类别标签,这里都用从0开始的自然数表示
    labels = kmeans.labels_
    assert len(labels) == len(X)
    # 聚类中心
    centers = kmeans.cluster_centers_
    assert len(centers) == k
    markers = ['o', '^', '*', 's']
    colors = ['r', 'b', 'y', 'k']
    # 对每一个类别
    for i in range(k):
        # 绘制该类对应的样本
        cluster = X[labels == i]
        plt.scatter(cluster[:, 0], cluster[:, 1], marker=markers[i], s=20, c=colors[i])
    # 绘制聚类中心点
    plt.scatter(centers[:, 0], centers[:, 1], marker='o', c='white', alpha=0.9, s=300)
    # 在中心点大白点(位置cnt)上绘制类别号i
    for i, cnt in enumerate(centers):
        plt.scatter(cnt[0], cnt[1], marker="$%d$" % i, s=50, c=colors[i])


if __name__ == '__main__':
    # 生成标准差为1的200个聚4类的2维样本点,聚类中心随机生成且每个维度都在-10到10的范围,最终将生成的两样本打乱
    X, y = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True,
                      random_state=1)
    # 聚类的类别数
    n_clusters = [2, 3, 4]
    plt.figure(figsize=(10, 3), dpi=100)
    # plt.xticks([])
    # plt.yticks(())
    for i, k in enumerate(n_clusters):
        plt.subplot(1, len(n_clusters), i + 1)
        fit_plot_kmeans_model(k, X)
    plt.show()

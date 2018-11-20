from sklearn.datasets.samples_generator import make_blobs  # 用于生成聚类样本
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # kNN分类

# 使matplotlib正常显示负号
plt.rcParams['axes.unicode_minus'] = False

'''
kNN分类,特征数=2(x轴坐标,y轴坐标),类别数目=3
'''

# 要生成的样本中心点
centers = [[-2, 2], [2, 2], [0, 4]]
# 围绕centers列表里提供的中心点坐标,以每类标准差0.6(可以传入列表指定不同方差),按随机种子0生成60个聚类样本
# 这里生成的X是样本的坐标,shape=(60,2);y是样本的类别标记,shape=(60,),因为是三类,其值只取0,1,2表示所属类别
X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.60)

# 训练(其实就是记下来)
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)

# 预测一个测试样本点,这里要转换成"样本数*特征数"shape的numpy数组
X_sample = np.array([0, 2]).reshape(-1, 2)
y_sample = clf.predict(X_sample)
print("预测样本点{}属于{}类".format(X_sample, y_sample))

# 看看预测它用到的最近的k个点
neighbors = clf.kneighbors(X_sample, return_distance=False)
print("它最近的{}个点在训练样本X中的索引是{}".format(k, neighbors))

# 绘制数据点
plt.figure(figsize=(16, 10), dpi=144)
c = np.array(centers)
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='orange')  # 中心点
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')  # 生成的聚类样本,这里用类别号当颜色号
plt.scatter(X_sample[:, 0], X_sample[:, 1], marker='x', c=y_sample, s=100, cmap='cool')  # 预测的样本
# 绘制最近的k个点到预测样本的连线
for i in neighbors[0]:  # 对于每个索引值i,表示它是X中行号为i的样本
    # 每次取k个最近邻点中的一个,和测试样本点放在一组里画线,就画出了k条从紧邻点到测试样本点的连线
    plt.plot([X[i, 0], X_sample[0, 0]], [X[i, 1], X_sample[0, 1]], 'k--', linewidth=0.6)
plt.show()

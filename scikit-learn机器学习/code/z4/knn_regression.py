import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt

# 使matplotlib正常显示负号
plt.rcParams['axes.unicode_minus'] = False

'''
kNN回归,特征数=1(x轴坐标),对每个x可以预测一个y值
'''

n_dots = 40
# 生成5*[0,1)之间的数据在数组中,shape=(n_dots,1)
X = 5 * np.random.rand(n_dots, 1)
# ravel()的作用和flatten一样,只是flatten()重新拷贝一份,但ravel()返回视图,操作ravel()后的会影响原始的
# np.cos(X).shape=(n_dots,1)和X一样;而np.cos(X).ravel().shape=(n_dots,)即是摊平以后的了
y = np.cos(X).ravel()
# 添加正负0.1范围内的噪声
y += 0.2 * np.random.rand(n_dots) - 0.1

# 训练回归模型
k = 5
knn = KNeighborsRegressor(k)
knn.fit(X, y)
print(knn.score(X, y))  # 得分0.98..,低的时候只有0.78..

# 生成足够密集的点,作为回归用的样本
# 这里[:, np.newaxis]表示为其添加一个维度,使其从shape=(500,)变成shape=(500,1)
T = np.linspace(0, 5, 500)[:, np.newaxis]
# 预测标签y的值,这里shape=(500,)
y_pred = knn.predict(T)

# 这些预测点构成拟合曲线
plt.figure(figsize=(16, 10), dpi=144)
plt.scatter(X, y, c='g', label="训练样本", s=50)  # s是点的大小
plt.plot(T, y_pred, c='k', label="预测样本构成拟合曲线", lw=2)  # lw就是linewidth,线宽
plt.axis("tight")  # 自动调整坐标轴使其能完整显示所有的数据
plt.title("kNN回归(k=%i)" % k)
plt.legend(loc="best")  # 自动选择最合适的位置设置图例
plt.show()

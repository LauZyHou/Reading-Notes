import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False

# 原始矩阵
A = np.array([
    [3, 2000],
    [2, 3000],
    [4, 5000],
    [5, 8000],
    [1, 2000]
])
# m行n列
m, n = A.shape

# 数据归一化:先计算每一列(特征)的均值,再用自动广播让每行(样本)减去均值
mean = np.mean(A, axis=0)
norm = A - mean

# 数据缩放:先计算每一列的最大最小值之差,再用自动广播让每一行除以这个差
scope = np.max(norm, axis=0) - np.min(norm, axis=0)
X = norm / scope

# 计算协方差矩阵Sigma(n阶方阵)
Sigma = (1.0 / m) * np.dot(X.T, X)

# 对协方差矩阵做奇异值分解
U, S, V = np.linalg.svd(Sigma)

# 因为是将二维(列)的数据变成一维(列),所以k=1
k = 1

# 取特征矩阵U(n阶方阵)的前k列构造主成分特征矩阵U_reduce(n行k列)
U_reduce = U[:, 0:k].reshape(n, k)

# 对数据进行降维,降维后的Z(m行k列)=X(m行n列)乘以U_reduce(n行k列)
Z = np.dot(X, U_reduce)
print(Z)

# 还原得到近似数据
X_approx = np.dot(Z, U_reduce.T)
A_approx = X_approx * scope + mean  # 这里书上用np.multiply也可以
print(A_approx)

# 绘图面板
plt.figure(figsize=(8, 8), dpi=100)
plt.title("降维及恢复示意图")
# 坐标最小最大值
mm = (-1, 1)
plt.xlim(*mm)
plt.ylim(*mm)
# 右/上坐标轴透明
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 绘制归一化之后的数据点
plt.scatter(X[:, 0], X[:, 1], marker='s', c='b')
# 绘制还原回来的数据点
plt.scatter(X_approx[:, 0], X_approx[:, 1], marker='o', c='r')
# 绘制主成分特征向量,即主成分特征矩阵U的n个列向量
plt.arrow(0, 0, U[0, 0], U[1, 0], color='r', linestyle='-')
plt.arrow(0, 0, U[0, 1], U[1, 1], color='r', linestyle='--')
# 接下来要在图上绘制标注,下面是提取出来的要给plt.annotate()用的一些参数
tit = [  # 标题
    r"$U_{reduce}=\mathbf{u}^{(1)}$",
    r"$\mathbf{u}^{(2)}$",
    "原始数据点",
    "投影点"
]
xy = [  # xy指定剪头尖端位置
    (U[0, 0], U[1, 0]),
    (U[0, 1], U[1, 1]),
    (X[0, 0], X[0, 1]),
    (X_approx[0, 0], X_approx[0, 1])
]
# xytest指定标注文本位置,用这个函数来从xy生成
pian = lambda a, b: (a + 0.2, b - 0.1)
# xycoords='data'表示使用轴坐标系,arrowprops指定箭头的属性.在这个嵌套dict中可以看到两种定义字典的写法
kw = {'xycoords': 'data', 'fontsize': 10, 'arrowprops': dict(arrowstyle="->", connectionstyle="arc3,rad=.2")}
# 绘制四个标注
for i in range(len(tit)):
    plt.annotate(tit[i], xy=xy[i], xytext=pian(*xy[i]), **kw)
plt.show()

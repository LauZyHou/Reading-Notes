import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error  # MSE损失
from matplotlib.figure import SubplotParams

# 使matplotlib正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 200个从-2pi到2pi的正弦函数样本点,上下波动0.1
n_dots = 200
X = np.linspace(-2 * np.pi, 2 * np.pi, n_dots)
y = np.sin(X) + 0.2 * np.random.randn(n_dots) - 0.1
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# 多项式回归模型
def polynomial_model(degree=1):
    # 多项式模型,指定多项式的次数和是否使用偏置(常数项)
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    # 线性回归模型,指定对每个特征归一化到(0,1)
    # (归一化只能提示算法收敛速度,不提高准确性)
    liner_regression = LinearRegression(normalize=True)
    # 装入管道
    pipline = Pipeline([("多项式", polynomial_features), ("线性回归", liner_regression)])
    return pipline


if __name__ == '__main__':
    degrees = [2, 3, 5, 10]
    models = []
    for d in degrees:
        model = polynomial_model(degree=d)
        # 这里会依次调用管道里的fit和transform(或者fit_transform),最后一个只调用fit
        model.fit(X, y)
        train_score = model.score(X, y)  # R2得分
        mse = mean_squared_error(y, model.predict(X))  # MSE损失
        print("degree:{}\tscore:{}\tmse loss:{}".format(d, train_score, mse))
        models.append({"model": model, "degree": d})  # 训练好的模型保存下来

    # 绘制不同degree的拟合结果,SubplotParams用于为子图设置统一参数,这里不用
    # plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=3.0))
    # fig, axes = plt.subplots(2, 2)
    for i, mod in enumerate(models):
        fig = plt.subplot(2, 2, i + 1)
        plt.xlim(-8, 8)
        plt.title("增加多项式特征的线性回归(次数={})".format(mod["degree"]))
        plt.scatter(X, y, s=5, c='b', alpha=0.5)
        plt.plot(X, mod["model"].predict(X), 'r-')
    # fig.tight_layout()
    plt.show()

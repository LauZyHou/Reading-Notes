import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot as plt

# 生成200个样本点
n_dots = 200
X = np.linspace(0, 1, n_dots)
# y=sqrt(X)上下浮动0.1
y = np.sqrt(X) + 0.2 * np.random.rand(n_dots) - 0.1

# sklearn里用的矩阵shape是(样本数,特征数)和(样本数,一维预测值)
# 这里正是200个1特征的样本,故转化为200x1的矩阵
X = X.reshape(-1, 1)  # shape:(200, 1)
y = y.reshape(-1, 1)  # shape:(200, 1)


# 多项式模型,传入多项式的次数
def polynomial_model(degree=1):
    # <class 'sklearn.preprocessing.data.PolynomialFeatures'>
    # 该类用于产生多项式的特征集
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    # <class 'sklearn.linear_model.base.LinearRegression'>
    # 该类用于创建线性回归的评估模型对象
    linear_regression = LinearRegression()
    # <class 'sklearn.pipeline.Pipeline'>
    # 装入流水线.每个元组第一个值为变量名,元组第二个元素是sklearn中的transformer或estimator
    # transformer即必须包含fit()和transform()方法,或者 fit_transform()
    # estimator即必须包含fit()方法
    # Pipeline.fit()可以对训练集进行训练,Pipeline.score()进行评分
    pipeline = Pipeline([("Polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    return pipeline


# 绘制模型的学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.title(title)
    # 设置y轴的刻度范围,这里用了序列解包
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("训练样本数")
    plt.ylabel("得分")
    # (5, 10) (5, 10)
    # 参数:
    # estimator=实现"拟合"和"测试"的模型,这里将Pipeline传入,X=数据集,y=样本,n_jobs=并行运行的作业数(-1表示使用全部核)
    # train_sizes=比例点数组,用于生成学习曲线的训练示例的相对或绝对数量(相对数量就是每百分之多少训练一次)
    # 返回值:
    # train_sizes_abs=每次用于生成学习曲线上的点的样本的数目.这里其shape=(5,),值=数组[16 52 88 124 160]
    # train_scores=在比例点拿样本训练,训练集的的得分.这里其shape=(5, 10),其列数=cv里的n_splits(划分次数)
    # test_scores=在比例点拿样本训练,测试集的的得分.这里其shape=(5, 10),其列数=cv里的n_splits(划分次数)
    train_sizes_abs, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                                train_sizes=train_sizes)
    # 做列的聚合,得到n_splits次训练-测试的方差和平均得分
    train_scores_mean = np.mean(train_scores, axis=1)  # shape=(5,),下同
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(True)  # 显示背景的网格线
    # plt.fill_between(x, y1, y2=0, where=None, interpolate=False, step=None, *, data=None, **kwargs)
    # 在两条水平曲线y1=f(x)和y2=g(x)之间填充颜色,这里是把模型训练和测试score的均值的上下方差的范围里填充半透明的红色和蓝色
    plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color='b')
    # 绘制score的均值点构成的曲线
    plt.plot(train_sizes_abs, train_scores_mean, 'o--', color='r', label="训练得分")
    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color='b', label="交叉验证得分")
    plt.legend(loc="best")  # 自动选择最合适的位置设置图例
    return plt


if __name__ == '__main__':
    # <class 'sklearn.model_selection._split.ShuffleSplit'>
    # 该对象用于将样本集合随机打乱后划分为训练集/测试集
    # n_splits=划分次数,test_size=测试集占比,train_size=训练集占比(默认None即取余下全部)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # 后面绘图用的三个子图的图例
    titles = ["学习曲线(欠拟合)", "学习曲线", "学习曲线(过拟合)"]
    # 三个多项式拟合模型的次数
    degrees = [1, 3, 10]

    plt.figure(figsize=(18, 4), dpi=200)

    # 在子图上分别画出三个次数的多项式的学习曲线
    for i in range(len(degrees)):
        plt.subplot(1, 3, i + 1)
        plot_learning_curve(polynomial_model(degrees[i]), titles[i], X, y, ylim=(0.75, 1.01), cv=cv)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 使matplotlib正常显示负号
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = "E:/WorkSpace/ReadingNotes/Python与机器学习实战"


# 训练样本,利用样本xs->ys来计算n次多项式系数ps
# 即获取使得MSE损失即[f(x,p)-y]^2最小的n次多项式系数p的序列
def train(xs, ys, n):
    ps = np.polyfit(xs, ys, n)
    assert len(ps) == n + 1  # 多项式系数从p0~pn,其数目一定是n+1
    return ps


# 以训练结果ps对指定的xs做预测并获得预测值
def predict(ps, xs):
    # 对每个x计算多项式SUM{ps[i]*x^(n-i)},i从0到n,即获得输入对应的预测值序列
    ys_ = np.polyval(ps, xs)
    return ys_


# 计算MSE损失,即SUM(0.5*[y-y']^2)
def get_mse_loss(ys, ys_):
    return 0.5 * ((ys - ys_) ** 2).sum()


if __name__ == '__main__':
    # 房子面积,房子价格
    xs, ys = [], []
    # 读入数据
    for line in open(BASE_DIR + "/data/z1/prices.txt"):
        x, y = line.split(",")
        xs.append(float(x))
        ys.append(float(y))
    xs, ys = np.array(xs), np.array(ys)
    # 横坐标(房子面积)标准化
    xs = (xs - xs.mean()) / xs.std()

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.scatter(xs, ys, c="k", s=20)  # c=点的颜色,s=点的大小

    # 生成拟合曲线的采样点
    x0s = np.linspace(-2, 4, 100)
    # 用于拟合数据点的多项式的系数n=1,n=4,n=10
    degs = (1, 4, 10)
    for d in degs:
        # 使用不同次的多项式做训练,并对采样点做预测以绘制拟合曲线
        ps = train(xs, ys, d)
        y0s_ = predict(ps, x0s)
        ax.plot(x0s, y0s_, label="多项式次数={}".format(d))
        # 计算在这个训练结果(ps)下原样本的MSE损失
        ys_ = predict(ps, xs)
        loss = get_mse_loss(ys, ys_)
        print("多项式次数={},MSE损失={}".format(d, loss))
    # 限制x轴和y轴范围
    plt.xlim(-2, 4)
    plt.ylim(1e5, 8e5)
    plt.legend()  # 用于显示图例
    plt.show()

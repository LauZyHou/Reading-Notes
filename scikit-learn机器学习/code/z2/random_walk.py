import matplotlib.pyplot as plt
import numpy as np

'''
一维随机漫步
'''

# 博弈组数
n_person = 2000
# 每组抛硬币次数
n_times = 500

# 抛硬币次数序列,用于当绘制点的横坐标
t = np.arange(n_times)
# 一共n_person组,每组是n_times个-1或者1这两个数组成的序列表示输和赢
# 即相当于创建0~1的整数(只有0和1),再*2-1也就是只有-1和1这两个数字组成的了
steps = 2 * np.random.random_integers(0, 1, (n_person, n_times)) - 1

# np.cumsum返回给定axis上的累计和
# 这里就是将二维steps的所有列逐步加起来的中间结果
# 这也是一个二维数组,反映了每组在博弈过程中逐步变化的的输赢总额
amounts = np.cumsum(steps, axis=1)
# 每个元素平方
sd_amount = amounts ** 2
# 再求所有行(组)的平均值
mean_sd_amount = sd_amount.mean(axis=0)

# Latex
plt.xlabel(r"$t$")
plt.ylabel(r"$\sqrt {\langle (\delta x)^2 \rangle}$")
# 绘制两条曲线
plt.plot(t, np.sqrt(mean_sd_amount), 'g.', t, np.sqrt(t), 'r-')
plt.show()

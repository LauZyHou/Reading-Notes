import numpy as np

'''
蒙特卡洛算法求π
'''

# 圆的半径设置为1,生成100W个点
n_dotes = 1000000
# random.random()生成0和1之间的随机浮点数
x = np.random.random(n_dotes)
y = np.random.random(n_dotes)
# 计算到圆心(原点)的距离.这里不对欧氏距离开方,后面只要用半径r^2=1比较就行
distance = x ** 2 + y ** 2
# 判断是否在圆内,使用布尔索引
in_circle = distance[distance < 1]

pi = 4 * float(len(in_circle)) / n_dotes
print(pi)

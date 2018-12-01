import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# 生成样本,y=2x噪声上下0.1
X_train = np.linspace(-1, 1, 100)
Y_train = 2 * X_train + np.random.randn(*X_train.shape) * 0.2 - 0.1
# plt.plot(X_train, y_train, 'ro', label="原始数据")
# plt.legend()
# plt.show()

"""建立模型"""
# 占位符
X = tf.placeholder("float")
Y = tf.placeholder("float")
# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")  # 一维的权重w_1,初始化为-1到1的随机数
b = tf.Variable(tf.zeros([1]), name="bias")  # 一维的偏置b,初始化为0
# (1)前向结构:通过正向生成一个结果
Z = tf.multiply(X, W) + b
# (2)定义损失的计算:这里就是所有的y和z的差的平方的平均值
# reduce_mean用于计算指定axis的平均值,未指定时则对tensor中每个数加起来求平均
cost = tf.reduce_mean(tf.square(Y - Z))
# (3)反向优化:通过反向过程调整模型参数.这里使用学习率为0.01的梯度下降最小化损失cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

"""迭代训练模型"""
# 初始化过程:初始化所有的变量
init = tf.global_variables_initializer()
# 模型的超参数,这里epoch是模型会完整学习多少次样本
train_epochs = 20
# 这个仅仅是控制每多少个epoch显示下模型的详细信息
display_step = 2

# 启动Session,这种方式不用手动关闭Session
with tf.Session() as sess:
    # 运行初始化过程
    sess.run(init)
    # 用于记录批次和损失
    plotdata = {"batchsize": [], "loss": []}
    # 向模型输入数据,对于每个epoch
    for epoch in range(train_epochs):
        # 都要遍历模型中所有的样本(x,y)
        for (x, y) in zip(X_train, Y_train):
            # 运行优化器,填充模型中X,Y占位符的内容为这个具体的样本(x,y)
            sess.run(optimizer, feed_dict={X: x, Y: y})
        # 每display_step个epoch显示一下训练中的详细信息
        if epoch % display_step == 0:
            # 计算下损失.每次要计算中间变量的当前值时候都要sess.run得到
            loss = sess.run(cost, feed_dict={X: X_train, Y: Y_train})
            print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))

            if not (loss == "NA"):


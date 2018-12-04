import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import math

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
cost = tf.reduce_mean(tf.square(Y - Z))  # <class 'tensorflow.python.framework.ops.Tensor'>
# (3)反向优化:通过反向过程调整模型参数.这里使用学习率为0.01的梯度下降最小化损失cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

"""迭代训练模型"""
# 初始化过程:初始化所有的变量
init = tf.global_variables_initializer()
# 模型的超参数,这里epoch是模型会完整学习多少次样本
train_epochs = 20
# 这个仅仅是控制每多少个epoch显示下模型的详细信息
display_step = 2
# tf.train.Saver用于保存模型到文件/从文件中取用模型
# 使用了检查点,这里指定max_to_keep=1即在迭代过程中只保存一个文件
# 那么在循环训练中,新生成的模型会覆盖之前的模型
saver = tf.train.Saver(max_to_keep=1)

# 启动Session,这种方式不用手动关闭Session
with tf.Session() as sess:
    # 运行初始化过程
    sess.run(init)
    # 用于记录批次和损失
    plotdata = {"ephch": [], "loss": []}
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
            print("Epoch:", epoch, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
            # 如果损失存在,将该批次和对应损失记录下来
            if loss != "NA":
                plotdata["ephch"].append(epoch)
                plotdata["loss"].append(loss)
            """保存检查点"""
            # 这里选择在每次输出信息后保存一下检查点,同时使用global_step记录epoch次数
            saver.save(sess, "./linermodel.ckpt", global_step=epoch)
    print("完成,cost=", sess.run(cost, feed_dict={X: X_train, Y: Y_train}), "W=", sess.run(W), "b=", sess.run(b))
    # 提前计算出结果以在Session外也能使用,下面这种方式都可以
    result = sess.run(W) * X_train + sess.run(b)
    result2 = sess.run(Z, feed_dict={X: X_train})
    # 这里验证一下它们中对应项的值是相等的
    for k1, k2 in zip(result, result2):
        assert math.isclose(k1, k2, rel_tol=1e-5)
    """使用模型"""
    # 如果要使用模型,直接将样本的值传入并计算输出Z即可
    print("对x=5的预测值:", sess.run(Z, feed_dict={X: 5}))
    """保存模型到文件"""
    # saver.save(sess, "./linermodel.ckpt")

"""训练模型可视化"""
plt.scatter(X_train, Y_train, c='r', marker='o', label="原始数据")
plt.plot(X_train, result, label="拟合直线")
plt.legend()
plt.show()


def moving_average(a, w=10):
    """
    对损失序列a,生成w-平均损失序列
    即每个位置的损失由其和其前的共w个损失来代替
    """
    if len(a) < w:  # 当w太小不足以计算任何元素的平均时
        return a[:]  # 直接返回a的复制
    return [val if idx < w else sum(a[idx - w:idx]) / w for idx, val in enumerate(a)]


"""绘制平均loss变化曲线"""
plotdata["avgloss"] = moving_average(plotdata["loss"])
plt.plot(plotdata["ephch"], plotdata["avgloss"], 'b--')
plt.xlabel("ephch")
plt.ylabel("avg loss")
plt.title("平均损失变化")
plt.show()

"""从文件载入模型"""
with tf.Session() as sess2:
    sess2.run(init)  # 还是需要运行一下初始化过程
    # saver.restore(sess2, "./linermodel.ckpt")
    # 在当前目录下寻找最近的检查点并载入
    ckpt = tf.train.latest_checkpoint("./")
    if ckpt is not None:
        saver.restore(sess2, ckpt)
    print("对x=5的预测值:", sess2.run(Z, feed_dict={X: 5}))

import tensorflow as tf
import numpy as np

"""生成样本"""
X_train = np.linspace(-1, 1, 100)
Y_train = 2 * X_train + np.random.randn(*X_train.shape) * 0.2 - 0.1

"""建立模型"""
X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

Z = tf.multiply(X, W) + b
tf.summary.histogram('z', Z)  # 预测值以直方图形式显示

cost = tf.reduce_mean(tf.square(Y - Z))
tf.summary.scalar('loss_function', cost)  # 损失以标量形式显示

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

"""迭代训练模型"""
init = tf.global_variables_initializer()

train_epochs = 20
display_step = 2

with tf.Session() as sess:
    sess.run(init)
    merged_summary_op = tf.summary.merge_all()  # 合并前面定义的所有的summary
    summary_writer = tf.summary.FileWriter('log/yeq2x', sess.graph)  # 用于写summary
    for epoch in range(train_epochs):
        for (x, y) in zip(X_train, Y_train):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        # 在每个epoch都生成summary
        summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
        # 然后将summary写入文件
        summary_writer.add_summary(summary_str, epoch)
    print("完成,cost=", sess.run(cost, feed_dict={X: X_train, Y: Y_train}), "W=", sess.run(W), "b=", sess.run(b))

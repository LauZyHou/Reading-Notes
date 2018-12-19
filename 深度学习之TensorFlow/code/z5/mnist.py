from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import pylab
import os

# 读取数据,将样本标签转换为one_hot编码(标签值位置的分量是1,其它位置是0)
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
print("训练数据shape:", mnist.train.images.shape)  # (55000, 784)即55000张28*28的图
print("测试数据shape:", mnist.test.images.shape)  # (10000, 784)
print("验证数据shape:", mnist.validation.images.shape)  # (5000, 784)

'''
# 显示第一张图看一下
im = mnist.train.images[0]
im = im.reshape(-1, 28)
pylab.imshow(im)
pylab.show()
'''

# 重置计算图
tf.reset_default_graph()
# 定义占位符,行(传入的样本数先不指定),而列数都是固定的
x = tf.placeholder(tf.float32, [None, 784])  # 对于特征就是784
y = tf.placeholder(tf.float32, [None, 10])  # 对于标签就是10

# 定义学习参数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义输出结点
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义交叉熵损失
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# 定义参数
learning_rate = 0.01
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

"""训练过程"""

# 定义训练相关的超参数
training_epochs = 25
batch_size = 100
display_step = 1

# 用于保存模型
saver = tf.train.Saver()
model_path = "log/model.ckpt"

# 启动session,并开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 对于每个epoch
    for epoch in range(training_epochs):
        avg_cost = 0.0
        # 因为"每次取batch_size个数据",这里可以计算一下一共有多少batch
        total_batch = int(mnist.train.num_examples / batch_size)
        # 都要遍历数据集中的所有数据
        for i in range(total_batch):
            # 每次取batch_size个数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行优化器
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # 计算平均损失
            avg_cost += c
        avg_cost /= total_batch
        # 每display_step个epoch结束时显示一些训练信息
        if (epoch + 1) % display_step == 0:
            print("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("训练完成")
    """测试过程"""
    # 返回one_hot编码中数值为1的元素下标,并依次进行比较即可
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算acc,即只要将True,False转换成1.0,0.0然后求平均值即可
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("acc=", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    """保存模型"""
    save_path = saver.save(sess, model_path)
    print("模型保存在:%s" % save_path)

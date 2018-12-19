import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pylab

"""仅仅是读取模型,数据不会存在模型里"""
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
"""读取之前定义的和训练时定义的图结构是一样的"""
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
"""读取模型"""
# 用于读取模型
saver = tf.train.Saver()
model_path = "log/model.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 恢复模型变量
    saver.restore(sess, model_path)

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率的结点
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # [真正进行计算的地方]
    print("acc=", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    # 预测结果序列
    output = tf.argmax(pred, 1)
    # 取训练集中的两个数据
    batch_xs, batch_ys = mnist.train.next_batch(2)
    # [真正进行计算的地方]计算一下预测结果及其one_hot形式
    outputval, predval = sess.run([output, pred], feed_dict={x: batch_xs})
    print(outputval, predval)

    # 输出这两个样本的图像看一下
    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()

import tensorflow as tf

# 定义模型略...
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    # 模型训练略...
    # 保存模型
    saver.save(sess, "mysave")

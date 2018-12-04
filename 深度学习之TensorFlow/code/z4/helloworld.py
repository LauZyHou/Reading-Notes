import tensorflow as tf

# 定义常量
hello = tf.constant("Hello ,TensorFlow!")
sess = tf.Session()
print(sess.run(hello))
sess.close()

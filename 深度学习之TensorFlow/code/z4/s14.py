import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 两个未命名的变量,TF会自动给名字
a = tf.Variable(1.0)
print("a:", a.name)
b = tf.Variable(2.0)
print("b:", b.name)
# 两个name一样的变量,TF会为第二个改名字
c = tf.Variable(3.0, name='var')
print("c:", c.name)
d = tf.Variable(4.0, name='var')
print("d:", d.name)

print("-" * 20)

# 在Session中读取变量的值
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("a:", a.eval())
    print("b:", b.eval())
    print("c:", c.eval())
    print("d:", d.eval())

with tf.variable_scope("v1"):
    ok = tf.get_variable("ok", [1], initializer=tf.constant_initializer(6.6))
    print("ok:", ok.name)

with tf.variable_scope("v2"):
    ok = tf.get_variable("ok", [1], initializer=tf.constant_initializer(6.6))
    print("ok:", ok.name)
'''
ok2 = tf.get_variable("ok")
print("ok2", ok2.name)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("ok:", ok.eval())
    print("ok2", ok2.eval())
'''
'''
with tf.variable_scope("test1"):
    a1 = tf.get_variable("Variable",[1])
    print(a1.name)
'''

with tf.variable_scope("v3", reuse=tf.AUTO_REUSE) as v3:
    p1 = tf.get_variable("p", [1], initializer=tf.constant_initializer(2.2))
    print("p1:", p1.name)
    p2 = tf.get_variable("p")
    print("p2:", p2.name)

# 查看下两个变量的值
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(p1), sess.run(p2))

with tf.variable_scope("v4"):
    ok4 = tf.get_variable("ok", [1], initializer=tf.constant_initializer(6.6))
    with tf.variable_scope("v5"):
        ok45 = tf.get_variable("ok", [1], initializer=tf.constant_initializer(6.6))

print("ok4:", ok4.name)
print("ok45:", ok45.name)

with tf.variable_scope("v6", reuse=tf.AUTO_REUSE):
    g1 = tf.get_variable("g1", [1], initializer=tf.constant_initializer(1.1))
    with tf.variable_scope("v7"):
        g2 = tf.get_variable("g2", [1], initializer=tf.constant_initializer(2.2))

with tf.variable_scope("v6", reuse=True):
    g3 = tf.get_variable("g1", [1], initializer=tf.constant_initializer(1.1))
    with tf.variable_scope("v7"):
        g4 = tf.get_variable("g2", [1], initializer=tf.constant_initializer(2.2))

print(g1.name, g3.name)
print(g2.name, g4.name)

with tf.variable_scope("s1", initializer=tf.constant_initializer(1.1)):
    s1a = tf.get_variable("s1a", [1])
    s1b = tf.get_variable("s1b", [1], initializer=tf.constant_initializer(1.2))
    with tf.variable_scope("s2"):
        s2a = tf.get_variable("s1a", [1])
    with tf.variable_scope("s3", initializer=tf.constant_initializer(2.1)):
        s3a = tf.get_variable("s1a", [1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(s1a), sess.run(s1b), sess.run(s2a), sess.run(s3a))

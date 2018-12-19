import tensorflow as tf

# 在TF默认的图上建立的常量Tensor
c = tf.constant(0.0)
print(c.graph)

# 建立图g,并在它上面建立个常量Tensor
g = tf.Graph()
with g.as_default():
    c1 = tf.constant(0.0)

c1_cpoy = g.as_graph_element(c1)
print(c1 is c1_cpoy)

print(c1.graph)  # 可以通过变量的graph属性获取所在的图
print(g)

# 获取默认图,看看默认图是哪个
g2 = tf.get_default_graph()
print(g2)

# 重置默认图,相当于重新建立了一个图
tf.reset_default_graph()  # 使用该函数时必须保证当前图的资源已经全部释放
g3 = tf.get_default_graph()
print(g3)

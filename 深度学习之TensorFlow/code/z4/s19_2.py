import tensorflow as tf

g = tf.Graph()
with g.as_default():
    c = tf.constant(0.0)
    d = tf.constant(1.1)

ops = g.get_operations()
print(ops)

print(c.name)
# 通过名称得到对应元素:通过Tensor的名称得到图中的c
t = g.get_tensor_by_name(name="Const:0")
print(c is t)

# 两个常量Tensor
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])
# 定义它们做矩阵乘法的操作
mymul = tf.matmul(a, b, name='mymul')
print(mymul.op.name)  # 注意这里是.op.name
# 因为这个op在默认图里,先获取到默认图
dft_g = tf.get_default_graph()
# 再从默认图里取出来
mymul_op = dft_g.get_operation_by_name(name="mymul")  # 注意这里没有':0'
mymul_tensor = dft_g.get_tensor_by_name(name="mymul:0")
print(mymul is mymul_op)
print(mymul_op is mymul_tensor)
print(mymul is mymul_tensor)

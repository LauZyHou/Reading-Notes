import tensorflow as tf

with tf.variable_scope("scopeA") as spA:
    var1 = tf.get_variable("v1", [1])

with tf.variable_scope("scopeB"):
    with tf.variable_scope(spA):
        var3 = tf.get_variable("v3", [1])

print(var3.name)

with tf.variable_scope("v"):
    with tf.name_scope("n1"):
        a = tf.get_variable("a", [1])  # Variable
        x = 1.0 + a  # Op
        with tf.name_scope(""):
            y = 1.0 + a  # Op
            b = tf.get_variable("b", [1])  # Variable仅受到variable_scope的限制
print(a.name, x.op.name, y.op.name, b.name, sep='\n')

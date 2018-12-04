from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

savedir = "../z3/"
# 默认情况下,Saver使用tf.Variable.name属性来保存变量
# 这里用print_tensors_in_checkpoint_file()输出其中所有变量和它的值
print_tensors_in_checkpoint_file(savedir + "linermodel.ckpt", tensor_name=None, all_tensor_names=True)

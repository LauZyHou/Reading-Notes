import tensorflow as tf

'''
按照时间保存检查点
'''

# 清除默认图堆栈,重置全局的默认图
tf.reset_default_graph()
# 创建(如果需要的话)并返回global step tensor,默认参数graph=None即使用默认图
# 当用这种方法按照时间保存检查点时,必须要定义这个global step张量
global_step = tf.train.get_or_create_global_step()
# 这里是为global_step张量每次增加1
step = tf.assign_add(global_step, 1)
# 通过MonitoredTrainingSession实现按时间自动保存,设置检查点保存目录,设置保存间隔为2秒
with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpoints', save_checkpoint_secs=2) as sess:
    # 输出一下当前global_step这个张量的值
    print(sess.run([global_step]))
    # 这里是一个死循环,只要sess不结束就一直循环
    while not sess.should_stop():
        # 运行step,即为global_step这个张量增加1
        i = sess.run(step)
        # 每次运行都输出一下
        print(i)

from sklearn import metrics
import numpy as np

# 随机序列
label_true = np.random.randint(1, 4, 6)
label_pred = np.random.randint(1, 4, 6)
print("Adjust Rand Index for 随机序列: %.3f" % metrics.adjusted_rand_score(label_true, label_pred))

# 结构相同,这个例子里能看出"对类别标签不敏感"
label_true = [1, 1, 3, 3, 2, 2]
label_pred = [3, 3, 2, 2, 1, 1]
print("Adjust Rand Index for 结构相同的序列: %.3f" % metrics.adjusted_rand_score(label_true, label_pred))

from sklearn import metrics
import numpy as np

label_true = [1, 1, 2, 2]
label_pred = [2, 2, 1, 1]
print("齐次性值 for 结构相同: %.3f" % metrics.homogeneity_score(label_true, label_pred))

label_true = [1, 1, 2, 2]
label_pred = [0, 1, 2, 3]
print("齐次性值 for 每个类别内只由一种原类别元素组成: %.3f" % metrics.homogeneity_score(label_true, label_pred))

label_true = [1, 1, 2, 2]
label_pred = [1, 2, 1, 2]
print("齐次性值 for 每个类别内不由一种原类别元素组成: %.3f" % metrics.homogeneity_score(label_true, label_pred))

label_true = np.random.randint(1, 4, 6)
label_pred = np.random.randint(1, 4, 6)
print("齐次性值 for 随机序列: %.3f" % metrics.homogeneity_score(label_true, label_pred))

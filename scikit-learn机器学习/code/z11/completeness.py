from sklearn import metrics
import numpy as np

label_true = [1, 1, 2, 2]
label_pred = [2, 2, 1, 1]
print("完整性值 for 结构相同: %.3f" % metrics.completeness_score(label_true, label_pred))

label_true = [0, 1, 2, 2]
label_pred = [2, 0, 1, 1]
print("完整性值 for 原类别相同的都分到一个聚类中: %.3f" % metrics.completeness_score(label_true, label_pred))

label_true = [0, 1, 2, 2]
label_pred = [2, 0, 1, 2]
print("完整性值 for 原类别相同的未分到一个聚类中: %.3f" % metrics.completeness_score(label_true, label_pred))

label_true = np.random.randint(1, 4, 6)
label_pred = np.random.randint(1, 4, 6)
print("完整性值 for 随机序列: %.3f" % metrics.completeness_score(label_true, label_pred))

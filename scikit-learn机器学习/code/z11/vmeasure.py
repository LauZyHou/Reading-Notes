from sklearn import metrics
import numpy as np

label_true = [1, 1, 2, 2]
label_pred = [2, 2, 1, 1]
print("V-measure for 结构相同: %.3f" % metrics.v_measure_score(label_true, label_pred))

label_true = [0, 1, 2, 3]
label_pred = [1, 1, 2, 2]
print("V-measure for 不齐次,但完整: %.3f" % metrics.v_measure_score(label_true, label_pred))
print("V-measure for 齐次,但不完整: %.3f" % metrics.v_measure_score(label_pred, label_true))

label_true = [1, 1, 2, 2]
label_pred = [1, 2, 1, 2]
print("V-measure for 既不齐次,也不完整: %.3f" % metrics.v_measure_score(label_true, label_pred))

label_true = np.random.randint(1, 4, 6)
label_pred = np.random.randint(1, 4, 6)
print("V-measure for 随机序列: %.3f" % metrics.v_measure_score(label_true, label_pred))

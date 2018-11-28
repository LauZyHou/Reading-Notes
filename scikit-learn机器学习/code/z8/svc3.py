from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from z8.svc2 import plot_hyperplane

X, y = make_blobs(n_samples=100, centers=3, random_state=0, cluster_std=0.8)
# 线性核
clf_linear = svm.SVC(C=1.0, kernel='linear')
# 多项式核
clf_poly = svm.SVC(C=1.0, kernel='poly', degree=3)
# 高斯核(RBF核)
clf_rbf = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
clf_rbf2 = svm.SVC(C=1.0, kernel='rbf', gamma=1.0)

plt.figure(figsize=(10, 10), dpi=144)

clfs = [clf_linear, clf_poly, clf_rbf, clf_rbf2]
titles = ["线性核", "多项式核", "高斯核$\gamma=0.5$", "高斯核$\gamma=1.0$"]

# 分别训练模型并绘制分类面.注意这里用zip组成一个个元组组成的对象
for clf, i in zip(clfs, range(len(titles))):
    clf.fit(X, y)
    plt.subplot(2, 2, i + 1)
    plot_hyperplane(clf, X, y, title=titles[i])
plt.show()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from z7.titanic import plot_curve  # 绘制score随参数变化的曲线
import numpy as np
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from z3.learning_curve import plot_learning_curve  # 绘制学习曲线
from sklearn.model_selection import ShuffleSplit

'''
SVM对乳腺癌数据集做分类
这里全部需要放在函数里调用,不然会来回调用这个文件自己(还不清楚为什么,这次不是文件重名的问题)
'''


def init():
    global X, y, X_train, y_train, X_test, y_test
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    print("shape:{},阳性样本数:{},阴性样本数{}".format(X.shape, y[y == 1].shape[0], y[y == 0].shape[0]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


if __name__ == '__main__':
    init()
    # 高斯核的SVM模型很复杂,在如此小的数据集上造成了过拟合
    clf = SVC(C=1.0, kernel='rbf', gamma=1.0)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print("train score:{},test score:{}".format(train_score, test_score))

    # 获取gamma参数在一个范围集合中的最优值
    gammas = np.linspace(0, 0.0003, 30)
    param_grid = {'gamma': gammas}
    clf = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True)
    clf.fit(X, y)
    print("最优参数:{},对应score:{}".format(clf.best_params_, clf.best_score_))
    # 绘制score随着参数变化的曲线
    plot_curve(gammas, clf.cv_results_, xlabel='gamma')
    plt.show()

    # 绘制学习曲线以观察模型拟合情况
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    title = "高斯核SVM的学习曲线"
    plot_learning_curve(SVC(C=1.0, kernel='rbf', gamma=0.0001), title, X, y, ylim=(0.5, 1.01), cv=cv)
    plt.show()

    # 使用二阶多项式核
    clf = SVC(C=1.0, kernel='poly', degree=2)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print("train score:{},test score:{}".format(train_score, test_score))

    # 对比一阶多项式和二阶多项式的学习曲线
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    title = "多项式核SVM的学习曲线,degree={}"
    degrees = [1, 2]
    plt.figure(figsize=(12, 4), dpi=144)
    for i in range(len(degrees)):
        plt.subplot(1, len(degrees), i + 1)
        # 这里n_jobs将传入learning_curve(),为并行运行的作业数
        plt = plot_learning_curve(SVC(C=1.0, kernel='poly', degree=degrees[i]), title.format(degrees[i]), X, y,
                                  ylim=(0.8, 1.01), cv=cv, n_jobs=4)
    plt.show()

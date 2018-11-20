from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import time
from z5.liner_fit_sin import polynomial_model
from z3.learning_curve import plot_learning_curve
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot as plt

# 读取boston房价数据集,并划分为训练集和测试集
boston = load_boston()
X = boston.data  # shape=(506,13)
y = boston.target  # shape=(506,)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# 使用前面写的增加了多项式特征的线性回归模型
model = polynomial_model(degree=2)  # 改成3时验证集上得分:-104.825038,说明过拟合
start = time.clock()  # 计时:训练和测试打分的用时
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
cv_score = model.score(X_test, y_test)
end = time.clock()
print("用时:{0:.6f},训练集上得分:{1:.6f},验证集上得分:{2:.6f}".format(end - start, train_score, cv_score))

# 绘制学习曲线
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plt.figure(figsize=(18, 4), dpi=100)
org_title = "学习曲线(degree={})"
degrees = (1, 2, 3)

for i in range(len(degrees)):
    plt.subplot(1, 3, i + 1)
    plt = plot_learning_curve(polynomial_model(degrees[i]), org_title.format(degrees[i]), X, y, ylim=(0.01, 1.01),
                              cv=cv)
plt.show()

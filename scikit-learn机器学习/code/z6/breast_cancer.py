from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import time
from z3.learning_curve import plot_learning_curve
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot as plt

'''
logistic回归:乳腺癌数据
'''

# 读取和划分数据
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print("X的shape={},正样本数:{},负样本数:{}".format(X.shape, y[y == 1].shape[0], y[y == 0].shape[0]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 查看模型得分
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("训练集得分:{trs:.6f},测试集得分:{tss:.6f}".format(trs=train_score, tss=test_score))

# 对测试集做预测
y_pred = model.predict(X_test)
# 这里作者书上有重大失误.正确的是这样:np.equal可以比较两个数组中的每一项,返回True/False数组
# 然后使用np.count_nonzero()统计其中True的数目,也就是预测正确的样本数
print("ACC:{}/{}".format(np.count_nonzero(np.equal(y_pred, y_test)), y_test.shape[0]))

# 找出预测概率低于0.9的样本:概率和为1,所以两个概率都>0.1时预测概率低于0.9
# 返回样本被预测为各类的概率
y_pred_proba = model.predict_proba(X_test)
# 类别号是0和1,这里得到的第一列应是预测为0的概率,第二列是预测为1的概率,这里用断言确保一下
assert y_pred[0] == (0 if y_pred_proba[0, 0] > y_pred_proba[0, 1] else 1)
# 全部样本中,预测为阴性的p>0.1的部分
y_pp_big = y_pred_proba[y_pred_proba[:, 0] > 0.1]
# 在这个基础上,找其预测为阳性的p>0.1的部分
y_pp_big = y_pp_big[y_pp_big[:, 1] > 0.1]
print(y_pp_big.shape)


# 用pipeline为logistic回归模型增加多项式特征
def polynomial_model(degree=2, **kwargs):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    logistic_regression = LogisticRegression(**kwargs)
    pipeline = Pipeline([("多项式特征", polynomial_features), ("logistic回归", logistic_regression)])
    return pipeline


# 增加多项式特征后的模型
# 这里使用L1范数做正则项,实现参数稀疏化(让某些参数减少到0)从而留下对模型有帮助的特征
model = polynomial_model(degree=2, penalty='l1')
start = time.clock()  # 计时:训练和测试用时
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
end = time.clock()
print("用时:{0:.6f},训练集上得分:{1:.6f},测试集上得分:{2:.6f}".format(end - start, train_score, test_score))

# 观察一下有多少特征没有因L1范数正则化项而被丢弃(减小到0)
# 从管道中取出logistic回归的estimator,使用加入管道时给出的名字
logistic_regression = model.named_steps["logistic回归"]
# coef_属性里保存的就是模型参数的值
print("参数shape:{},其中非0项数目:{}".format(logistic_regression.coef_.shape, np.count_nonzero(logistic_regression.coef_)))

# 绘制新模型的学习曲线
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
org_titlt = "增加{}阶多项式特征和{}正则化项的logistic回归学习曲线"
degrees = [1, 2]
penaltys = ["l1", "l2"]
fig = plt.figure(figsize=(12, 10), dpi=100)
for p in range(len(penaltys)):
    for i in range(len(degrees)):
        plt.subplot(len(penaltys), len(degrees), p * len(degrees) + i + 1)
        plt = plot_learning_curve(polynomial_model(degree=degrees[i], penalty=penaltys[p]),
                                  org_titlt.format(degrees[i], penaltys[p]), X, y, ylim=(0.8, 1.01), cv=cv)
fig.tight_layout()
plt.show()

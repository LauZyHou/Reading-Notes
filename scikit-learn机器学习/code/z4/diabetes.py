import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier  # kNN和半径kNN
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from z3.learning_curve import plot_learning_curve  # 上节实践的绘制学习曲线
from sklearn.feature_selection import SelectKBest  # 特征选择

'''
kNN和RN预测印第安人糖尿病
'''

BASE_DIR = "E:/WorkSpace/ReadingNotes/scikit-learn机器学习/data/"

# 读取数据,路径中有中文的时候要用open打开再给pandas读取
with open(BASE_DIR + "z4/diabetes.csv") as f:
    df = pd.read_csv(f, header=0)  # shape=(768, 9),其中前8个是特征,最后一列是标签

# 分离特征(前8列)和标签(第9列)
X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

# 模型列表,里面存("模型名称",模型对象)
models = []
k = 2
# 普通kNN模型
models.append(("kNN", KNeighborsClassifier(n_neighbors=k)))
# 带权重的kNN:距离作为权重,距离越近的话语权越重
models.append(("带权重的kNN", KNeighborsClassifier(n_neighbors=k, weights="distance")))
# RN(这里指基于半径的邻居算法)
models.append(("半径kNN", RadiusNeighborsClassifier(n_neighbors=k, radius=500.0)))

# ----------------------------------------------

print("-" * 20 + "一次划分和训练" + "-" * 20)
# 分离训练集和测试集:拿出其中20%的样本作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 训练三个模型,计算在测试集上的score
for name, model in models:
    model.fit(X_train, y_train)
    print(name, ":", model.score(X_test, y_test))

# ----------------------------------------------

print("-" * 20 + "多次划分和训练,取平均" + "-" * 20)
# K折交叉验证对象,这里表示划分10次(K=10)
kfold = KFold(n_splits=10)
# 对于每个模型
for name, model in models:
    # 用K折交叉验证的方式对样本集做划分,并训练得到K个score
    scores = cross_val_score(model, X, y, cv=kfold)
    print(name, ":", scores.mean())  # 取平均

# ----------------------------------------------

# 训练
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# 查看在训练集和测试集上的score
train_score = knn.score(X_train, y_train)  # 0.8420195439739414
test_score = knn.score(X_test, y_test)  # 0.6753246753246753
print("训练集score:{},测试集score:{}".format(train_score, test_score))

# 绘制学习曲线
knn = KNeighborsClassifier(n_neighbors=2)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)  # 划分10份,测试集占20%
plt.figure(figsize=(10, 6), dpi=200)
# 这里要把函数里面最后一句return plt打开,全局的多余代码注释掉
plt = plot_learning_curve(knn, "kNN(k=2)学习曲线", X, y, ylim=(0.0, 1.01), cv=cv)
plt.show()

# ----------------------------------------------

# 特征选择
selector = SelectKBest(k=2)  # 只选择其中两个和输出相关性最大的
X_new = selector.fit_transform(X, y)  # 这里会寻找和y相关性最大的X中的特征,提取相应列的数据

# 如果只使用这两个相关性最大的特征,比较前述的三种模型平均score
print("-" * 20 + "只使用两个相关性最大的特征" + "-" * 20)
for name, model in models:
    scores = cross_val_score(model, X_new, y, cv=kfold)
    print(name, ":", scores.mean())

# ----------------------------------------------

# 观察数据的分布情况,只使用刚刚提取的两个特征,这样就可以可视化在平面上
plt.figure(figsize=(10, 6), dpi=200)
plt.xlabel("血糖浓度")
plt.ylabel("BMI指数")
# 分别绘制阴性样本(y=0)和阳性样本(y=1)
# 数组和数值的比较运算可以返回每个数值进行比较运算的True/False序列
# 然后这里是投给X做布尔索引,也就返回了相应的样本数据
plt.scatter(X_new[y == 0][:, 0], X_new[y == 0][:, 1], c='r', s=20, marker='o')
plt.scatter(X_new[y == 1][:, 0], X_new[y == 1][:, 1], c='b', s=20, marker='^')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

digits = datasets.load_digits()

# train_test_split函数用于将矩阵随机划分为训练子集和测试子集,并返回划分好的训练/测试样本,训练/测试标签
# test_size=:0~1之间的浮点数:测试样本占比;整数:测试样本的数量.random_state:随机数的种子
Xtrain, Xtest, Ytrain, Ytest = train_test_split(digits.data, digits.target, test_size=0.20, random_state=2)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)  # (1437, 64) (360, 64) (1437,) (360,)

# 使用C-SVM训练模型,C是对离群点的惩罚因子,gamma是核的系数,默认用RBF核
clf = svm.SVC(C=100.0, gamma=0.001)
clf.fit(Xtrain, Ytrain)
# 在测试集上做测试,返回决定系数R2=1-(SSE/SST),其中SSE是残差平方和,SST是总平方和
r2 = clf.score(Xtest, Ytest)
print(r2)  # 0.9777777777777777

# 获取在测试集上预测的具体结果:使用clf.predict(测试样本)即可
Ypred = clf.predict(Xtest)
# 创建子图图形.接下来绘制前16个测试样本的测试结果
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
# 调整子图外围的边距
fig.subplots_adjust(hspace=0.1, wspace=0.1)
# numpy.flat返回数组摊平为一维的结果,用这种方式方便地遍历axes中的每个ax
for i, ax in enumerate(axes.flat):
    # 训练和测试的数据都是摊平后的,这里要调整成8*8的图像再输入
    ax.imshow(Xtest[i].reshape(8, 8), cmap='gray_r', interpolation='nearest')
    # 这里transform=ax.transAxes使前面的位置坐标参数是相对于左下角坐标轴原点,而不是默认的右上角
    ax.text(0.05, 0.05, str(Ypred[i]), fontsize=32, transform=ax.transAxes,
            color='green' if Ypred[i] == Ytest[i] else 'red')  # 左下角:预测值
    ax.text(0.8, 0.05, str(Ytest[i]), fontsize=32, transform=ax.transAxes, color='black')  # 右下角:真实值
    ax.set_xticks(())  # 不显示刻度
    ax.set_yticks(())
plt.show()

# 保存训练好的模型
joblib.dump(clf, "../../data/z2/digits_svm.pkl")

# 读取保存下来的训练好的模型
# clf = joblib.load("../../data/z2/digits_svm.pkl")
# print(clf.score(Xtest, Ytest))

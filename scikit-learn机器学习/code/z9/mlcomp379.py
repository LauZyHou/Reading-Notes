from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

# 读取数据,这里的数据文件每一类在一个子目录中,子目录的名字即是文档的类别
news_train = load_files(r"E:\Data\code\datasets\mlcomp\379\train")
print("共{}文档分为{}类".format(len(news_train.data), len(news_train.target_names)))
# data中存的是所有文档,target中存的是类别号,若要知道类别名,在target_names中查询
print("0号文档的类别名:", news_train.target_names[news_train.target[0]])

# 转化为由TF-IDF表达的权重信息构成的向量
vectorizer = TfidfVectorizer(encoding='latin-1')
# 相当于先fit(完成语料分析,提取词典),再transform(把每篇文档转化为向量以构成矩阵)
X_train = vectorizer.fit_transform((d for d in news_train.data))
y_train = news_train.target
print("样本数:%d,特征数%d" % X_train.shape)
print("样本{}中的非零特征数为{}".format(news_train.filenames[0], X_train[0].getnnz()))

# 多项式分布的朴素贝叶斯.其中alpha是平滑参数,越小越容易造成过拟合
clf = MultinomialNB(alpha=0.001)
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
print("训练集得分:", train_score)

# 加载测试集
news_test = load_files(r"E:\Data\code\datasets\mlcomp\379\test")
print("共{}文档分为{}类".format(len(news_train.data), len(news_train.target_names)))

# 对测试集进行向量化,前面的语料分析和提取词典是基于训练集的,这里只调用transform
X_test = vectorizer.transform((d for d in news_test.data))
y_test = news_test.target
print("样本数:%d,特征数%d" % X_test.shape)

# 尝试预测第一篇文档
pred = clf.predict(X_test[0])
print("预测为:{},实际是:{}".format(pred[0], news_test.target[0]))

# 在整个测试集上做预测
pred = clf.predict(X_test)
# 查看对每个类别的预测准确性
print("使用的分类器是", clf, "分类表现如下")
print(classification_report(y_test, pred, target_names=news_test.target_names))

# 生成混淆矩阵
cm = confusion_matrix(y_test, pred)
print("混淆矩阵:\n", cm)

# 混淆矩阵可视化
plt.figure(figsize=(8, 8), dpi=144)
plt.title("混淆矩阵")
ax = plt.gca()  # Get Current Axes
ax.spines['right'].set_color('none')  # ax.spines是数据区域的边界
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.xaxis.set_ticks_position('none')  # 删除轴上的刻度点
ax.yaxis.set_ticks_position('none')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.matshow(cm, fignum=1, cmap='gray')  # plt.matshow专门用于矩阵可视化
plt.colorbar()  # 添加渐变色条
plt.show()

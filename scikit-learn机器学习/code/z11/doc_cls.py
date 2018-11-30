from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

# 读取数据
docs = load_files("E:\Data\code\datasets\clustering\data")
data = docs.data
target_names = docs.target_names
print("summary: {} documents in {} categories.".format(len(data), len(target_names)))

# 生成词典并将文档转化为TF-IDF向量
# 在生成词典时,过滤超过max_df比例(或数目)或者min_df比例(或数目)的词,最大保留20000个特征,编码ISO-8859-1
vectorizer = TfidfVectorizer(max_df=0.4, min_df=2, max_features=20000, encoding='latin-1')
X = vectorizer.fit_transform(data)
print("n_samples:{},n_features:{}".format(*X.shape))
print("0号样本中的非零特征数目:", X[0].getnnz())

k = 4
# 参数n_init即是多次选取不同的初始化聚类中心,而最终输出的是score最大(成本最小)的聚类
# 参数max_iter指定一次KMeans过程中最大的循环次数,即便聚类中心还可以移动,到达这个最大次数也结束
# 参数tol决定中心点移动距离小于多少时认定算法已经收敛
kmeans = KMeans(n_clusters=k, max_iter=100, tol=0.01, verbose=0, n_init=3)
kmeans.fit(X)
print("k={},cost={}".format(k, "%.3f" % kmeans.inertia_))

# 查看1000~1009这10个文档的聚类结果及其本来的文件名,可以看一下目录一样的也就是同一类的
print(kmeans.labels_[1000:1010])
print(docs.filenames[1000:1010])

# 查看每种类别文档中,影响最大(即那个维度数值最大)的10个单词
# 对得到的每个聚类中心点进行排序得到排序索引,这里"::-1"使其按照从大到小排序
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
# 显然越排在前面的对应的索引值所对应的单词影响越大
# 取出词典中的词
terms = vectorizer.get_feature_names()
# 对每个聚类结果i
for i in range(k):
    print("Cluster %d" % i, end='')
    # 取出第i行(也就是第i个聚类中心点)前10重要的词的索引
    for ind in order_centroids[i, :10]:
        # 在词典term中可以按这个索引拿到对应的词
        print(" %s" % terms[ind], end='')
    print()

# 评价聚类表现
label_true = docs.target  # 标记的类别
label_pred = kmeans.labels_  # 聚类得到的类别
print("齐次性: %.3f" % metrics.homogeneity_score(label_true, label_pred))
print("完整性: %.3f" % metrics.completeness_score(label_true, label_pred))
print("V-measure: %.3f" % metrics.v_measure_score(label_true, label_pred))
print("Adjust Rand Index: %.3f" % metrics.adjusted_rand_score(label_true, label_pred))
print("轮廓系数: %.3f" % metrics.silhouette_score(X, label_pred, sample_size=1000))

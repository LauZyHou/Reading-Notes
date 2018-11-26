import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np

BASE_DIR = "E:/WorkSpace/ReadingNotes/scikit-learn机器学习/data/"


# 读取泰坦尼克号数据,并做一定的预处理
def read_dataset(file):
    # index_col指定作为行索引的列,这里第一列是PassengerId
    df = pd.read_csv(file, index_col=0)
    # 丢弃无用的特征(指定axis=1即列),inplace=True则在df对象上操作,而不是返回操作后的df
    df.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
    # 将性别转换为男1女0:先转换为True/False序列再进行类型转换
    df['Sex'] = (df['Sex'] == 'male').astype('int')
    # 将登船港口数据转化为数值型数据,先获得其中的所有可能取值放在列表中,再直接取其在列表中的下标即可
    embarked_unique = df['Embarked'].unique().tolist()
    df['Embarked'] = df['Embarked'].apply(lambda x: embarked_unique.index(x))
    # 缺失数据(NaN)设置为0
    df = df.fillna(0)
    return df


if __name__ == '__main__':
    with open(BASE_DIR + "z7/train.csv") as f:
        df = read_dataset(f)
    # 标签即"是否存活"一列
    y = df['Survived'].values
    # 特征里要去掉标签这一列
    X = df.drop(['Survived'], axis=1).values
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_impurity_decrease=0.0, min_samples_split=22)
    clf.fit(X, y)
    print(clf.score(X, y))
    # 读入测试集并做预测,然后保存结果
    with open(BASE_DIR + "z7/test.csv") as f2:
        test_df = read_dataset(f2)
    predictions = clf.predict(test_df.values)
    # 转换成要求的格式
    result = pd.DataFrame({'PassengerId': test_df.index, 'Survived': predictions.astype(np.int32)})
    # print(result)
    result.to_csv("./titanic.csv", index=False)

import pandas as pd
import numpy as np

'''
data = {'col1': {'wr': 30, 'wr1': 20, 'wr2': 50},
        'col2': {'wr': 100, 'wr1': 90, 'wr2': 80},
        'col3': {'wr': 91, 'wr1': 90, 'wr2': 80}}
df = pd.DataFrame(data)
print(df)
df = df.loc['wr'][df.loc['wr'] > 90].index.values
print(df)
'''
'''
data = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(data.argsort())
'''

"""
pandas传入sklearn的处理测试
"""

x = np.random.randint(0, 2, [100, 50], np.int8)
dfx = pd.DataFrame(x)
# print(dfx.shape)
print(type(dfx))
# y = np.zeros([100, ], np.int8)
dfy = dfx[dfx.columns[-1]]
# print(dfy.shape)
print(type(dfy))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.2)
print(type(X_train), type(X_test), type(y_train), type(y_test))

from sklearn.svm import SVC

print(type(y_test), y_test.shape)
clf = SVC()
clf.fit(X_train, y_train)
ok = clf.predict(X_test)
print(type(ok), ok.shape)

print(ok)
print(y_test.values)

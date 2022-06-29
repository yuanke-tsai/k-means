# k-means

Language: Python

## 1. k-means 是什麼？

 k-means 是一個集區分類方式。透過指派一個中心點（質心）作為集區根據地，計算資料離哪一個集區最近，則該筆資料屬於離他最近的集區。

缺點：當資料維度很大時，易停在局部最佳解。

## 2. 計算舉例

- 表一為初始質心，透過表一與資料點表二，計算出距離（表格灰色處為計算結果），並且指派x1~x5分別距離z1~z3誰最近。
- 透過分類資料，把相同分類的劃分在一起且計算這些相同分類的資料平均值（表三），做為新的質心點（表四）。
- 用新的質心點繼續下一迭代（重複表二~表四），直到達到停止條件（停止條件可自己設定）。

表一：質心

|  | feature1 | feature2 |
| --- | --- | --- |
| z1 | -1 | 0 |
| z2 | 2 | 8 |
| z3 | 6 | 0 |

表二：資料要做分類看是屬於 z1~z3 哪一集區

|  | feature1 | feature2 | z1點距離 | z2點距離 | z3點距離 | min |
| --- | --- | --- | --- | --- | --- | --- |
| x1 | -1 | 1 | 1.000 | 7.616 | 7.071 | z1 |
| x2 | 3 | 8 | 8.944 | 1.000 | 8.544 | z2 |
| x3 | 0 | 0 | 1.000 | 8.246 | 6.000 | z1 |
| x4 | 4 | 9 | 10.296 | 2.236 | 9.220 | z2 |
| x5 | 5 | 0 | 6.000 | 8.544 | 1.000 | z3 |

表三：做質心的更新計算

| z1更新 | feature1 | feature2 |
| --- | --- | --- |
| x1 | -1 | 1 |
| x3 | 0 | 0 |
| average | -0.5 | 0.5 |

| z2更新 | feature1 | feature2 |
| --- | --- | --- |
| x2 | 3 | 8 |
| x4 | 4 | 9 |
| average | 4.5 | 8.5 |

| z3更新 | feature1 | feature2 |
| --- | --- | --- |
| x5 | 5 | 0 |
|  |  |  |
| average | 5 | 0 |

表四：更新後的質心

|  | feature1 | feature2 |
| --- | --- | --- |
| z1 | -0.5 | 0.5 |
| z2 | 4.5 | 8.5 |
| z3 | 5 | 0 |

## 3. code 說明（未用 k-means 套件）

以下範例使用公開資料 iris 做訓練。由於k-means需要先決定有幾叢，所以我們先將資料做簡單了解。從下圖中可以大概看出k=3（三群）。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
Y = iris.target

print(iris.feature_names)
plt.scatter(X[:,0:1], X[:,1:2],c="green")
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.title("Distribution of IRIS from Length")
plt.show()
```

![Output from Code1](k-means%20a9992470eadf4712a15ba50e2f9986a3/k-means_plot.jpg)

Output from Code1

接下來就是一連串的迭代囉，這裡停止條件設比較簡單，跑到更新點為目前最大，且與前一次迭代相同即停止。

```python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:51:06 2022

@author: user
"""
# 沒有要建預測模型，因此全部一起跑
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy import stats

data = datasets.load_iris()
X = data.data
Y = data.target
k = 3
np.random.seed(1)

# 初始設定，隨機選三筆資料作為初始質心(只使用一次)
idx = np.random.choice(150, size=3)
centroid = {}
for i in range(k):
    centroid[i] = X[idx[i]]
print(centroid)
    
# 歐式距離計算
def Euclidean(X, k, centroid):
    minDistance = np.array([])
    for x in range(len(X)):
        distanceList = np.array([])
        for i in range(k):
            distance = np.sqrt(sum(np.square(centroid.get(i) - X[x])))
            distanceList = np.append(distanceList, distance)
        minDistance = np.append(minDistance, np.argmin(distanceList)).astype(int)
    
    # 根據離質心距離最近的分類在一起，並組成 dataframe
    df = pd.DataFrame(zip(X, minDistance, Y))
    df.rename(columns = {0: 'DataX', 1: 'yPlum', 2: 'Y'}, inplace = True)
    return df

def UpdateCentroid(df, X, Y, k):
    # Group 組成來源檢視
    group = df.groupby("yPlum")
    # print(group)
    # for key, item in group:
    #     print(group.get_group(key), "\n\n")  # 一行行檢視 group 的狀況
    modeNoList = np.array([])
    centroid = {}
    for i in range(k):
        yLabel = group.get_group(i)
        y = stats.mode(yLabel["Y"])[0][0] # stats.mode 傳出的資料為陣列
        newLabel = Y[yLabel.index]
        modeNo = newLabel.tolist().count(y)
        modeNoList = np.append(modeNoList, modeNo, axis=None)
        poolAccuracy =(modeNo / len(yLabel))
        Xcentroid = sum(yLabel['DataX'])/len(yLabel)
        centroid[i] = Xcentroid
        # print('第 {} 來自屬於 {}: 集區正確率: {}'.format(i, y, poolAccuracy))
    
    Accuracy = (sum(modeNoList) / len(X))
    print('\n整體正確率: {}'.format(Accuracy))
    # print(centroid)
    return centroid, Accuracy

Accuracy = np.array([])
i = 0
while True:
    df = Euclidean(X, k, centroid)
    centroid, accuracy = UpdateCentroid(df, X, Y, k)
    Accuracy = np.append(Accuracy, accuracy, axis=None)
    if len(Accuracy) == 1:
        continue
    elif (Accuracy[i+1] == max(Accuracy)) & (Accuracy[i] == Accuracy[i+1]):
        break
    i += 1
```
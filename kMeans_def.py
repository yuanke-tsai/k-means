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
# print(centroid)
    
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
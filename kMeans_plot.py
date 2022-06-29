# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:38:04 2022

@author: user
"""

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

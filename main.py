from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer,load_digits
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np


digitsData = load_digits()
x = digitsData.data
y = digitsData.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
"""cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(x_train,c=y_train,marker="o",s=40,hist_kwds={'bins':15}, figsize=(9,9),cmap=cmap)"""
neighbors = np.arange(1,15)
train_acc = np.empty(len(neighbors))
test_acc = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    train_acc[i] = knn.score(x_train, y_train)
    test_acc[i] = knn.score(x_test, y_test)
    print(f"Test Accuracy {test_acc[i]}")
    print(f"Train Accuracy {train_acc[i]}")
    print("====----====----====----====----====")

plt.plot(neighbors,test_acc,label="Testing Dataset Accuracy")
plt.legend()
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.show()

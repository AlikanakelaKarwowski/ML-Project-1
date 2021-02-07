from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer,load_digits
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats

#Load the data into an object
breastData = load_breast_cancer()

#Split up the data
x = breastData.data
y = breastData.target

#Split The data into training and testing arrays
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.75, random_state=20)

#Set np arrays for training, testing, and neighbors
neighbors = np.arange(1,50)
train_acc = np.empty(len(neighbors))
test_acc = np.empty(len(neighbors))

#run through testing for each k number of neighbors
for i,k in enumerate(neighbors):
    #training
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    #Save accuracy for both training and testing
    train_acc[i] = knn.score(x_train, y_train)
    test_acc[i] = knn.score(x_test, y_test)

#Show Relevant plots based on requirements
plt.plot(neighbors,test_acc,label="Testing Dataset Accuracy")
plt.legend()
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.show()



#Show Relavent data based on requirements
unique, counts = np.unique(breastData.target,return_counts=True)
classification = dict(zip(breastData.target_names, counts))
print(f'Features and Attributes (Total): {len(breastData.feature_names)}')
print(f'Features and Attributes (Type): {breastData.feature_names}')
print(f'Number of Classes: {unique}')
print(f'Distribution of Classes: {classification}')
print(f'Dataset Partition (Training): 75.5%')
print(f'Dataset Partition (Testing): 24.5%')
print(f'Distance Metric: {knn.effective_metric_}')
print(f'Estimate Accuracy at 5 Nearest Neighbors: {round(test_acc[3],3)}')
print(f'Estimate Accuracy at 15 Nearest Neighbors: {round(test_acc[13],3)}')
print(f'Estimate Accuracy at 30 Nearest Neighbors: {round(test_acc[28],3)}')
print(f'Estimate Accuracy at 50 Nearest Neighbors: {round(test_acc[48],3)}')
print(f'Min: {round(test_acc.min(),3)}')
print(f'Max: {round(test_acc.max(),3)}')
print(f'Mean: {round(test_acc.mean(),3)}')
print(f'Median: {round(np.median(test_acc),3)}')
print(f'Mode: {round(float(stats.mode(test_acc)[0]),3)}')
print(f'STD: {round(np.std(test_acc),3)}')


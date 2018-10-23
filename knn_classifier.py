"""
Created on Tue Oct 23 23:02:26 2018
@author: ruhul_amin_raju
"""

import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
#print(type(iris))

X = iris.data[:, :2]
y = iris.target
k = 7
#print(type(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=12)

train_set_size = len(X_train)
test_set_size = len(X_test)

y_pred = np.zeros(test_set_size)

distances = np.zeros(X_train.shape[0])

for i in range(test_set_size):
    x1 = X_test[i, :]
    for j in range(train_set_size):
        x2 = X_train[j, :]
        distances[j] = math.sqrt(np.sum((x1-x2)**2))
    
    index = np.argsort(distances)[:k]
    markers = np.zeros(len(np.unique(y_train)))
    
    for j in range(k):
        markers[y_train[index[j]]] = markers[y_train[index[j]]] + 1
    
    p_class = np.argmax(markers)
    y_pred[i] = p_class


number_of_correctly_classified_test_instances = np.count_nonzero(y_pred==y_test)
print('Accuracy:', (number_of_correctly_classified_test_instances/test_set_size) * 100, '%')

#from sklearn.metrics import accuracy_score
#print('Accuracy:', accuracy_score(y_test, y_pred) * 100, '%')
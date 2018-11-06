"""
Created on Wed Oct 24 23:02:26 2018
@author: ruhul_amin_raju
"""

import numpy as np
import math
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

iris = load_diabetes()

X = iris.data
y = iris.target
K_list = [20,21,22,23,24]
MSE = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=6)

train_set_size = len(X_train)
test_set_size = len(X_test)

y_pred = np.zeros(test_set_size)

distances = np.zeros(X_train.shape[0])

for k in K_list:
    for i in range(test_set_size):
        x1 = X_test[i]
        for j in range(train_set_size):
            x2 = X_train[j]
            distances[j] = math.sqrt(np.sum((x1-x2)**2))
            
        knn = np.argsort(distances)[:k]
        #markers = np.zeros(len(np.unique(y_train)))
            
        cval = 0
        for j in range(k):
            cval += y_train[knn[j]]
                
        cval /= k
        y_pred[i] = cval
        
    mse = np.sum((y_pred-y_test)**2)
    mse /= len(y_test)
    MSE.append(mse)
    
    print('K = {}, MSE = {}'.format(k, mse))

best_K = np.argmin(MSE)
print('\nBest choice for K is: {}'.format(K_list[best_K]))

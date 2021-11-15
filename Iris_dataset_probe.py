# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:58:52 2021

@author: G531
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np 
from sklearn.model_selection import StratifiedKFold

X, y = load_iris(return_X_y=True)
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)
K = 5
test_error_rate = np.empty((K,1))

k=0
for train_index, test_index in skf.split(X, y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]

    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    mu = np.mean(X_train, 0)
    sigma = np.std(X_train, 0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    mdl = LogisticRegression(multi_class='multinomial',
                                      penalty='l2', random_state=(1),
                                      C=1/100, 
                                      max_iter= 5000)
    mdl.fit(X_train,y_train)
    y_test_est = mdl.predict(X_test)
    test_error_rate[:,k] = np.sum(y_test_est!=y_test) / len(y_test)


# clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
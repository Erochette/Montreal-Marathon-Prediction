from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cross_validation import train_test_split

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import ProcessImage
import csv
import theano


def load_dataset():
    std_scaler = StandardScaler()
    train_y = np.genfromtxt('train_y.csv', delimiter=',', dtype='int32', skip_header=1)[:, 1]
    X = np.fromfile('train_x.bin',dtype='uint8')
    train_x = X.reshape((-1,60*60))
    test_x2=X.reshape((-1,60*60))
    train_x=train_x[:80000]
    test_x2=test_x2[80001:99999]
    train_y = np.genfromtxt('train_y.csv', delimiter=',', dtype='int32', skip_header=1)[:, 1]

    pca = RandomizedPCA(n_components=60)
    
    train_x=pca.fit_transform(train_x)
    test_x2 = pca.transform(test_x2)
    train_x = std_scaler.fit_transform(train_x)
    test_x2 = std_scaler.transform(test_x2)
    train_y=train_y[:80000]
    test_y2=train_y[800001:99999]

    return train_x,train_y,test_x2, test_y2
print "loading data.."
X, y,test_x2,test_y2= load_dataset()
print "Data loaded"
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, y)
print test_y2
res= clf.predict(test_x2)
correct = 0
total = 0
i=0
for value in res:
    if value==test_y2.item(i):
        correct+=1
    total+=1
    i+=1
print correct
print total
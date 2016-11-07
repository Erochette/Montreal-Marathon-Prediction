from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import ProcessImage
import csv
import theano


def load_dataset():
    std_scaler = StandardScaler()
    train_x = np.fromfile('train_x.bin',dtype='uint8')
    train_x = train_x.reshape((100000,60*60))
    test_x2=train_x[800001:99999]
    
    pca = RandomizedPCA(n_components=60)

    train_x=train_x[:80000]
    train_x=pca.fit_transform(train_x)
    test_x2 = pca.transform(test_x2)
    train_x = std_scaler.fit_transform(train_x)
    test_x2 = std_scaler.transform(test_x2)
    train_y = np.genfromtxt('train_y.csv', delimiter=',', dtype='int32', skip_header=1)[:, 1]
    test_y2=train_y[80001:99999]
    train_y=train_y[:80000]

    return train_x,train_y,test_x2, test_y2
print "loading data.."
X, y,test_x2,test_y2= load_dataset()
print "Data loaded"
clf = KNeighborsClassifier(n_neighbors=500)
clf.fit(X, y)
print "done"
print "="*20
#print clf

print "Confusion Matrix"
print "="*40
#print confusion_matrix(test_y2, clf.predict(test_x2))
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
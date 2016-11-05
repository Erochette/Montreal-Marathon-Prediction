from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from lasagne.layers import get_all_params
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from sklearn.cross_validation import train_test_split

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import ProcessImage
import csv
import theano

def load_dataset(subsetsize=None):
    train_x = np.fromfile('train_x.bin',dtype='uint8')
    train_x = train_x.reshape((100000,1,60, 60))
    #train_x= ProcessImage.process_images(train_x)
    #train_x = train_x.reshape(-1,1,60,60)
    test_x = np.fromfile('test_x.bin', dtype='uint8')
    test_x = test_x.reshape((-1,1, 60, 60))
    test_x = ProcessImage.process_images(test_x)

    train_y = np.genfromtxt('train_y.csv', delimiter=',', dtype='int32', skip_header=1)[:, 1]
    #train_y = train_y.flatten()
    train_x = np.array(train_x).astype(np.float32)
    train_y = np.array(train_y).astype(np.int32)
    test_x = np.array(test_x).astype(np.float32)
    # return theano.shared(np.array(train_x, dtype=theano.config.floatX).reshape(-1, 60, 60)), \
    #        theano.shared(np.asarray(train_y, dtype=theano.config.floatX)).flatten(), \
    #        theano.shared(np.array(test_x, dtype=theano.config.floatX).reshape(-1, 60, 60))
    return train_x,train_y,test_x
print "loading data.."
X, y,test_x= load_dataset()
print "Data loaded"
# figs, axes = plt.subplots(4, 4, figsize=(6, 6))
# for i in range(4):
#     for j in range(4):
#         axes[i, j].imshow(-X[i + 4 * j].reshape(60, 60), cmap='gray', interpolation='none')
#         axes[i, j].set_xticks([])
#         axes[i, j].set_yticks([])
#         axes[i, j].set_title("Label: {}".format(y[i + 4 * j]))
#         axes[i, j].axis('off')
layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2],X.shape[3])}),

    # first stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # # second stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    # the output layer
    (DenseLayer, {'num_units': 19, 'nonlinearity': softmax}),
]
net0 = NeuralNet(
    layers=layers0,
    max_epochs=200,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    #objective_l2=0.0025,

    train_split=TrainSplit(eval_size=0.25),
    verbose=1,
)
print "starting to fit data"
print X.shape
print y.shape
net0.fit(X, y)
print "I finished fiting"
results= net0.predict(test_x)
print "I finished predicting"
with open('CNNresultsWithOpt.csv','wb') as csvfile: #save for later
    writer=csv.writer(csvfile)
    for value in results:
        writer.writerow(value)
print "Completely done"

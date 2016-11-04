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

def load_dataset(subsetsize=None):

    subset = True
    if subsetsize is None:
        subset = False
        subsetsize = 100000

    train_x = np.fromfile('train_x.bin', dtype='uint8')
    train_x= ProcessImage.process_images(train_x)
    train_x = train_x.reshape((100000, 60, 60))
    test_x = np.fromfile('test_x.bin', dtype='uint8')
    test_x = ProcessImage.process_images(test_x)
    test_x = test_x.reshape((20000, 60, 60))

    train_y = np.genfromtxt('train_y.csv', delimiter=',', dtype='int32', skip_header=1)[:, 1]

    if subset:
        train_x = train_x[:subsetsize]
        train_y = train_y[:subsetsize]

    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2)

    val_size = subsetsize*0.2
    train_size = subsetsize - val_size

    return theano.shared(np.array(X_train, dtype=theano.config.floatX).reshape(train_size, 60, 60)), \
           theano.shared(np.asarray(y_train, dtype=theano.config.floatX)).flatten(), \
           theano.shared(np.array(X_val, dtype=theano.config.floatX).reshape(val_size, 60, 60)), \
           theano.shared(np.asarray(y_val, dtype=theano.config.floatX)).flatten(), \
           theano.shared(np.array(test_x, dtype=theano.config.floatX).reshape(20000, 60, 60))

    # We first define a download function, supporting both Python 2 and 3.
    # with open(path, 'rb') as f:
    #     next(f)  # skip header
    #     for line in f:
    #         idNum, yi = line.split(',', 1)
    #         y.append(yi)

    # # if sys.version_info[0] == 2:
    # #     from urllib import urlretrieve
    # # else:
    # #     from urllib.request import urlretrieve

    # # def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    # #     print("Downloading %s" % filename)
    # #     urlretrieve(source + filename, filename)

    # # We then define functions for loading MNIST images and labels.
    # # For convenience, they also download the requested files if needed.
    # import gzip

    # def load_mnist_images(filename):
    #     if not os.path.exists(filename):
    #         download(filename)
    #     # Read the inputs in Yann LeCun's binary format.
    #     with gzip.open(filename, 'rb') as f:
    #         data = np.frombuffer(f.read(), np.uint8, offset=16)
    #     # The inputs are vectors now, we reshape them to monochrome 2D images,
    #     # following the shape convention: (examples, channels, rows, columns)
    #     data = data.reshape(-1, 1, 60, 60)
    #     # The inputs come as bytes, we convert them to float32 in range [0,1].
    #     # (Actually to range [0, 255/256], for compatibility to the version
    #     # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    #     return data / np.float32(256)

    # def load_mnist_labels(filename):
    #     if not os.path.exists(filename):
    #         download(filename)
    #     # Read the labels in Yann LeCun's binary format.
    #     with gzip.open(filename, 'rb') as f:
    #         data = np.frombuffer(f.read(), np.uint8, offset=8)
    #     # The labels are vectors of integers now, that's exactly what we want.
    #     return data

    # # We can now download and read the training and test set images and labels.
    # X = load_mnist_images('train-images-idx3-ubyte.gz')
    # y = load_mnist_labels('train-labels-idx1-ubyte.gz')
    # # Theano works with fp32 precision
    # train_x=[]
    # train_y=[]
    # test_x=[]
    # y_val=[]

    # train_x = np.fromfile('train_x.bin', dtype='uint8')
    # train_x = train_x.reshape((100000, 60, 60))

    # test_x = np.fromfile('test_x.bin', dtype='uint8')
    # test_x = test_x.reshape((20000, 3600))

    # train_y = np.genfromtxt('train_y.csv', delimiter=',', skip_header=1)[:, 1]

    # if subset:
    #     train_x = train_x[:subsetsize]
    #     train_y = train_y[:subsetsize]

    # print("pre-processing images")
    # train_x = ProcessImage.process_images(train_x)

    # X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2)

    # val_size = subsetsize*0.2
    # train_size = subsetsize - val_size

    # X_train=np.array(X_train).reshape((train_size, 60, 60))
    # y_train=np.array(y_train, dtype='uint8')
    # X_val=np.array(X_val).reshape((val_size, 60, 60))
    # y_val=np.array(y_val, dtype='uint8')
    
    # X_train = np.array(X_train).astype(np.float32)
    # y_train = np.array(y_train).astype(np.int32)

    # # apply some very simple normalization to the data
    # # X -= X.mean()
    # # X /= X.std()

    # # For convolutional layers, the default shape of data is bc01,
    # # i.e. batch size x color channels x image dimension 1 x image dimension 2.
    # # Therefore, we reshape the X data to -1, 1, 28, 28.
    # X_train = X_train.reshape(
    #     -1,  # number of samples, -1 makes it so that this number is determined automatically
    #     1,   # 1 color channel, since images are only black and white
    #     60,  # first image dimension (vertical)
    #     60 ,  # second image dimension (horizontal)
    # )

    # return X_train, y_train,y_val, X_val,x_test
    # here you should enter the path to your MNIST data
path = os.path.join(os.path.expanduser('~'), 'Documents/lasagne/train.csv')
print "loading data.."
X, y,y_val,X_val,x_test = load_dataset()
print "Data loaded"
figs, axes = plt.subplots(4, 4, figsize=(6, 6))
for i in range(4):
    for j in range(4):
        axes[i, j].imshow(-X[i + 4 * j].reshape(60, 60), cmap='gray', interpolation='none')
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        axes[i, j].set_title("Label: {}".format(y[i + 4 * j]))
        axes[i, j].axis('off')
layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

    # first stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # second stage of our convolutional layers
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
    (MaxPool2DLayer, {'pool_size': 2}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    # the output layer
    (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]
net0 = NeuralNet(
    layers=layers0,
    max_epochs=200,

    update=adam,
    update_learning_rate=0.0002,

    objective_l2=0.0025,

    train_split=TrainSplit(eval_size=0.25),
    verbose=1,
)
print "starting to fit data"
net0.fit(X, y)
print "I finished fiting"
results= net0.predict(test_x)
print "I finished predicting"
with open('CNNresults.csv','wb') as csvfile: #save for later
    writer=csv.writer(csvfile)
    for value in results:
        writer.writerow(value)
print "Completely done"
#plot_loss(net0)
# layers1 = [
#     (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

#     (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3)}),
#     (MaxPool2DLayer, {'pool_size': (2, 2)}),

#     (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3)}),
#     (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3)}),
#     (MaxPool2DLayer, {'pool_size': (2, 2)}),

#     (Conv2DLayer, {'num_filters': 96, 'filter_size': (3, 3)}),
#     (MaxPool2DLayer, {'pool_size': (2, 2)}),

#     (DenseLayer, {'num_units': 64}),
#     (DropoutLayer, {}),
#     (DenseLayer, {'num_units': 64}),

#     (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
# ]
# net1 = NeuralNet(
#     layers=layers1,
#     update_learning_rate=0.01,
#     verbose=2,
# )
# net1.initialize()
# layer_info = PrintLayerInfo()
# layer_info(net1)
# layers4 = [
#     (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

#     (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
#     (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
#     (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
#     (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
#     (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
#     (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
#     (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
#     (MaxPool2DLayer, {'pool_size': (2, 2)}),

#     (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
#     (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
#     (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
#     (MaxPool2DLayer, {'pool_size': (2, 2)}),

#     (DenseLayer, {'num_units': 64}),
#     (DropoutLayer, {}),
#     (DenseLayer, {'num_units': 64}),

#     (DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
# ]
# net4 = NeuralNet(
#     layers=layers4,
#     update_learning_rate=0.01,
#     verbose=2,
# )
# net4.initialize()
# layer_info(net4)
# net4.verbose = 3
# layer_info(net4)
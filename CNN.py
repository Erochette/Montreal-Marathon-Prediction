from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne.visualize import draw_to_notebook
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from nolearn.lasagne.visualize import plot_saliency
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot
import pydotplus as pydot
import matplotlib.cm as cm
import numpy
import ProcessImage
import csv
import time


def write_prediction(list_of_results, index):
    print("Writing Prediciton file")
    with open('predictions_cnn0_%s_%s.csv' % (index, time.strftime("%Y%m%d-%H%M%S")), 'ab') as output_file:
        header = ['Id', 'Prediction']
        writer = csv.DictWriter(output_file, fieldnames=header)
        writer.writeheader()
        for j, prediction in enumerate(list_of_results):
            writer.writerow({'Id': j, 'Prediction': prediction})

    print "Completely done"


def graph_stats(network, X_valid, y_valid):
    # This prints graph of train and valid loss
    train_loss = numpy.array([j["train_loss"] for j in network.train_history_])
    valid_loss = numpy.array([j["valid_loss"] for j in net.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()

    # saves architecture to notebook
    draw_to_notebook(network)

    # we need to define a
    print("predicting...")
    results = net0.predict(X_valid)
    print("done")
    cm = confusion_matrix(y_valid, results)
    pyplot.matshow(cm)
    pyplot.title('Confusion matrix')
    pyplot.colorbar()
    pyplot.ylabel('Actual label')
    pyplot.xlabel('Predicted label')
    pyplot.show()


def run_epochs(network, X_in, y_in, test_in, numb_of_epochs):
    for i in xrange(numb_of_epochs):
        network.fit(X_in, y_in)
        network.save_weights_to('model.npz')

        print "I finished fitting"
        results = network.predict(test_in)
        print "I finished predicting"
        write_prediction(results, i)


if __name__ == '__main__':

    print "loading data.."
    X, y, test_x = ProcessImage.load_dataset_cnn()
    print "Data loaded"

    layers0 = [
        # layer dealing with the input data
        (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

        # first stage of our convolutional layers
        (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
        (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
        (Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
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

    layers4 = [
        (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

        (Conv2DLayer, {'num_filters': 32, 'filter_size': (5, 5), 'pad': 1}),
        (Conv2DLayer, {'num_filters': 32, 'filter_size': (5, 5), 'pad': 1}),
        (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
        (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
        (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
        (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
        (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (Conv2DLayer, {'num_filters': 128, 'filter_size': (5, 5), 'pad': 1}),
        (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1}),
        (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 1}),
        (MaxPool2DLayer, {'pool_size': (2, 2)}),

        (DenseLayer, {'num_units': 64}),
        (DropoutLayer, {}),
        (DenseLayer, {'num_units': 64}),

        (DenseLayer, {'num_units': 19, 'nonlinearity': softmax}),
    ]

    net4 = NeuralNet(
        layers=layers4,
        update_learning_rate=0.05,
        verbose=2,
        update_momentum=0.9,
        train_split=TrainSplit(eval_size=0.2),
        max_epochs=20,
    )

    net0 = NeuralNet(
        layers=layers0,
        max_epochs=20,

        # update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        # objective_l2=0.0025,

        train_split=TrainSplit(eval_size=0.25),
        verbose=1,
    )
    print "starting to fit data"
    print X.shape
    print y.shape

    # Uncomment to run a previous model
    # net0.load_weights_from('model.npz')
    # net0.initialize()

    # alternative network to run
    # net = net4
    net = net0

    run_epochs(net, X, y, test_x, 500)
    graph_stats(net, X, y)






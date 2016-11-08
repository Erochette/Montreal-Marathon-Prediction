import json
import random
import sys
import csv
import time
import warnings
import numpy as np
from sklearn.cross_validation import train_test_split


class NeuralNetClassifier:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.initialize_weights()

    def initialize_weights(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def predict(self, X, name=None):
        """
        return list of predictions after training algorithm
        """
        print("Writing Prediciton file")
        with open('predictions%s_%s.csv' % (name, time.strftime("%Y%m%d-%H%M%S")), 'ab') as output_file:
            header = ['Id', 'Prediction']
            writer = csv.DictWriter(output_file, fieldnames=header)
            writer.writeheader()
            for i, prediction in enumerate(X):
                writer.writerow({'Id': i, 'Prediction': np.argmax(self.feedforward(prediction))})

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def fit(self, training_data, epochs, mini_batch_size, momentum,
            regularization = 0.0,
            evaluation_data=None):
        """Train the neural network using mini-batch stochastic gradient
        descent with momentum and regularization"""
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for i, mini_batch in enumerate(mini_batches):
                self.update_mini_batch(("%s / %s" % (i, len(mini_batches))),
                                       mini_batch, momentum, regularization, len(training_data))

            self.save('state.json')
            data = []
            data_dict = {}
            print(" ")
            print("Epoch %s training complete" % j)

            cost = self.total_cost(training_data, regularization)
            training_cost.append(cost)
            print("Cost on training data: {}".format(cost))
            train_cost = {'Training_Cost': cost}
            data_dict.update(train_cost)

            accuracy = self.accuracy(training_data)
            training_accuracy.append(accuracy)
            print("Accuracy on training data: {} / {}".format(
                accuracy, n))
            train_acc = {'Training_Accuracy': accuracy}
            data_dict.update(train_acc)

            cost = self.total_cost(evaluation_data, regularization)
            evaluation_cost.append(cost)
            print("Cost on evaluation data: {}".format(cost))
            valid_cost = {'Validation_Cost': cost}
            data_dict.update(valid_cost)

            accuracy = self.accuracy(evaluation_data)
            evaluation_accuracy.append(accuracy)
            print("Accuracy on evaluation data: {} / {}".format(
                self.accuracy(evaluation_data), n_data))
            valid_acc = {'Validation_Accuracy': accuracy}
            data_dict.update(valid_acc)
            print(" ")
            data.append(data_dict)
            self.save_data(data)
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, index, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for i, (x, y) in enumerate(mini_batch):
            sys.stdout.write("\rminibatch %s processing:%s/%s" % (index, i, len(mini_batch)))
            sys.stdout.flush()
            delta_nabla_b, delta_nabla_w = self.forward_and_back_pass(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def forward_and_back_pass(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # forward pass
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = w.dot(activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).deriv_cross_entropy(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(activations[-2].T)
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = self.weights[-l+1].T.dot(delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = delta.dot(activations[-l-1].T)
        return (nabla_b, nabla_w)

    def accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                    for (x, y) in data]
        return sum(int(x == int(y)) for (x, y) in results)

    def total_cost(self, data, lmbda):
        cost = 0.0
        predictions = [self.feedforward(x) for (x, y) in data]
        for i, (x, y) in enumerate(data):
            a = predictions[i]
            cost += self.cost.cross_entropy(a, y) / len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def save_data(self, data):
        keys = data[0].keys()
        with open('data.csv', 'ab') as data_file:
            dict_writer = csv.DictWriter(data_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)
        data_file.close()


def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = NeuralNetClassifier(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def sigmoid(z):
    """The sigmoid function."""
    # Different variants of sigmoid
    # return 1.0/(1.0+np.exp(-z))
    # return expit(z)
    # Approximation of sigmoid
    return z / (1 + abs(z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def cross_entropy(a, y):
    """Cross entropy cost function"""
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


def deriv_cross_entropy(z, a, y):
    """Derivative of cross entropy cost function"""
    return (a-y)


def one_hot_encode(labels):
    """One Hot Encode the labels"""
    n_values = np.max(labels) + 1
    return np.eye(n_values)[labels]


def load_dataset(subsetsize=None):

    subset = True
    if subsetsize is None:
        subset = False
        subsetsize = 100000

    train_x = np.fromfile('processed_train_x_small.bin', dtype='uint8')
    train_x = train_x.reshape((100000, 30, 30))

    test_x = np.fromfile('processed_test_x_small.bin', dtype='uint8')
    test_x = test_x.reshape((20000, 30, 30))

    train_y = np.genfromtxt('train_y.csv', delimiter=',', skip_header=1)[:, 1]

    if subset:
        train_x = train_x[:subsetsize]
        train_y = train_y[:subsetsize]

    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2)

    val_size = subsetsize*0.2
    train_size = subsetsize - val_size

    return np.array(X_train).reshape((train_size, 900, 1)), np.array(y_train, dtype='int64'), np.array(X_val).reshape(
        (val_size, 900, 1)), np.array(y_val, dtype='int64'), np.array(test_x).reshape(20000, 900, 1)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # use load_dataset(1000) to only use 1000 of the training set
    X_train, y_train, X_valid, y_valid, X_test = load_dataset()

    y_train = one_hot_encode(y_train)
    y_valid = one_hot_encode(y_valid)

    train = [(x, np.expand_dims(y, axis=1)) for x, y in zip(X_train, y_train)]

    validate = [(x, np.expand_dims(y, axis=1)) for x, y in zip(X_valid, y_valid)]

    print("Building NN Classifier")
    # nn = load('state.json')
    nn = NeuralNetClassifier([900, 4000, 19])
    print("Fitting Data")
    nn.fit(train, 10, 1000, .9, 0.01, validate, True, True, True, True)

    test = [(x) for x in X_test]
    # pred_train = [(x) for x in X_train]
    # pred_valid = [(x) for x in X_valid]
    # print("Making Predictions on Training Set")
    # predictions = nn.predict(pred_train, "_train")
    # print("Making Predictions on Validation Set")
    # predictions = nn.predict(pred_valid, "_valid")
    print("Making Predictions on Test Set")
    predictions = nn.predict(test, "_realtest")





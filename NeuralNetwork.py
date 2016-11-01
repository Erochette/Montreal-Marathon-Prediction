import random
import numpy as np
import ProcessImage
from sklearn.cross_validation import train_test_split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


# using softmax as output layer is recommended for classification where outputs are mutually exclusive
def softmax(w):
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist


# using tanh over logistic sigmoid for the hidden layer is recommended
def tanh(x):
    return np.tanh(x)


# derivative for tanh sigmoid
def dtanh(y):
    return 1 - y * y


def relu(x):
    return max(0, x)


def d_relu(y):
    return 1 if y > 0 else 0


def one_hot_encode(labels):
    n_values = np.max(labels) + 1
    return np.eye(n_values)[labels]


def load_dataset(subsetsize=None):

    subset = True
    if subsetsize is None:
        subset = False
        subsetsize = 100000

    train_x = np.fromfile('train_x.bin', dtype='uint8')
    train_x = train_x.reshape((100000, 60, 60))

    test_x = np.fromfile('test_x.bin', dtype='uint8')
    test_x = test_x.reshape((20000, 3600))

    train_y = np.genfromtxt('train_y.csv', delimiter=',', skip_header=1)[:, 1]

    if subset:
        train_x = train_x[:subsetsize]
        train_y = train_y[:subsetsize]

    print("pre-processing images")
    train_x = ProcessImage.process_images(train_x)

    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2)

    val_size = subsetsize*0.2
    train_size = subsetsize - val_size

    return np.array(X_train).reshape((train_size, 60, 60)), np.array(y_train, dtype='uint8'), np.array(X_val).reshape(
        (val_size, 60, 60)), np.array(y_val, dtype='uint8'), test_x


class MLP_Classifier(object):
    """
    Basic MultiLayer Perceptron (MLP) neural network with regularization and learning rate decay
    Consists of three layers: input, hidden and output. The sizes of input and output must match data
    the size of hidden is user defined when initializing the network.
    The algorithm can be used on any dataset.
    As long as the data is in this format: [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]],
                                           ...
                                           [[[x1, x2, x3, ..., xn], [y1, y2, ..., yn]]]
    An example is provided below with the digit recognition dataset provided by sklearn
    Fully pypy compatible.
    """

    def __init__(self, input, hidden, output, iterations=50, learning_rate=0.01,
                 l2_in=0, l2_out=0, momentum=0, rate_decay=0,
                 output_layer='logistic', verbose=True):
        """
        :param input: number of input neurons
        :param hidden: number of hidden neurons
        :param output: number of output neurons
        :param iterations: how many epochs
        :param learning_rate: initial learning rate
        :param l2: L2 regularization term
        :param momentum: momentum
        :param rate_decay: how much to decrease learning rate by on each iteration (epoch)
        :param output_layer: activation (transfer) function of the output layer
        :param verbose: whether to spit out error rates while training
        """
        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.l2_in = l2_in
        self.l2_out = l2_out
        self.momentum = momentum
        self.rate_decay = rate_decay
        self.verbose = verbose
        self.output_activation = output_layer

        # initialize arrays
        self.input = input + 1  # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # set up array of 1s for activations
        self.ai = np.ones(self.input)
        self.ah = np.ones(self.hidden)
        self.ao = np.ones(self.output)

        # create randomized weights
        # use scheme from Efficient Backprop by LeCun 1998 to initialize weights for hidden layer
        input_range = 1.0 / self.input ** (1 / 2)
        self.wi = np.random.normal(loc=0, scale=input_range, size=(self.input, self.hidden))
        self.wo = np.random.uniform(size=(self.hidden, self.output)) / np.sqrt(self.hidden)

        # create arrays of 0 for changes
        # this is essentially an array of temporary values that gets updated at each iteration
        # based on how much the weights need to change in the following iteration
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        """
        The feedforward algorithm loops over all the nodes in the hidden layer and
        adds together all the outputs from the input layer * their weights
        the output of each node is the sigmoid function of the sum of all inputs
        which is then passed on to the next layer.
        :param inputs: input data
        :return: updated activation output vector
        """
        if len(inputs) != self.input - 1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # input activations
        self.ai[0:self.input - 1] = inputs

        # hidden activations
        sum = np.dot(self.wi.T, self.ai)
        self.ah = tanh(sum)

        # output activations
        sum = np.dot(self.wo.T, self.ah)
        if self.output_activation == 'logistic':
            self.ao = sigmoid(sum)
        elif self.output_activation == 'softmax':
            self.ao = softmax(sum)
        else:
            raise ValueError('Choose a compatible output layer activation or check your spelling ;-p')

        return self.ao

    def backPropagate(self, targets):
        """
        For the output layer
        1. Calculates the difference between output value and target value
        2. Get the derivative (slope) of the sigmoid function in order to determine how much the weights need to change
        3. update the weights for every node based on the learning rate and sig derivative
        For the hidden layer
        1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
        2. get derivative to determine how much weights need to change
        3. change the weights based on learning rate and derivative
        :param targets: y values
        :param N: learning rate
        :return: updated weights
        """
        if len(targets) != self.output:
            raise ValueError('Wrong number of targets you silly goose!')

        # calculate error terms for output
        # the delta (theta) tell you which direction to change the weights
        if self.output_activation == 'logistic':
            output_deltas = dsigmoid(self.ao) * -(targets - self.ao)
        elif self.output_activation == 'softmax':
            output_deltas = -(targets - self.ao)
        else:
            raise ValueError('Choose a compatible output layer activation or check your spelling ;-p')

            # calculate error terms for hidden
        # delta (theta) tells you which direction to change the weights
        error = np.dot(self.wo, output_deltas)
        hidden_deltas = dtanh(self.ah) * error

        # update the weights connecting hidden to output, change == partial derivative
        change = output_deltas * np.reshape(self.ah, (self.ah.shape[0], 1))
        regularization = self.l2_out * self.wo
        self.wo -= self.learning_rate * (change + regularization) + self.co * self.momentum
        self.co = change

        # update the weights connecting input to hidden, change == partial derivative
        change = hidden_deltas * np.reshape(self.ai, (self.ai.shape[0], 1))
        regularization = self.l2_in * self.wi
        self.wi -= self.learning_rate * (change + regularization) + self.ci * self.momentum
        self.ci = change

        # calculate error
        if self.output_activation == 'softmax':
            error = -sum(targets * np.log(self.ao))
        elif self.output_activation == 'logistic':
            error = sum(0.5 * (targets - self.ao) ** 2)

        return error

    def test(self, patterns):
        """
        Currently this will print out the targets next to the predictions.
        Not useful for actual ML, just for visual inspection.
        """
        for p in patterns:
            print(p[1], '->', self.feedForward(p[0]))

    def fit(self, patterns):
        if self.verbose == True:
            if self.output_activation == 'softmax':
                print('Using softmax activation in output layer')
            elif self.output_activation == 'logistic':
                print('Using logistic sigmoid activation in output layer')

        num_example = np.shape(patterns)[0]

        for i in range(self.iterations):
            error = 0.0
            random.shuffle(patterns)
            for i, p in enumerate(patterns):
                print("fitting input: %s" % i)
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)

            with open('error.txt', 'a') as errorfile:
                errorfile.write(str(error) + '\n')
                errorfile.close()

            if i % 10 == 0 and self.verbose == True:
                error = error / num_example
                print('Training error %-.5f' % error)

            # learning rate decay
            self.learning_rate = self.learning_rate * (
            self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))

    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions



if __name__ == "__main__":
    train_X, train_y, val_X, val_y, test_X = load_dataset()

    train_y = one_hot_encode(train_y)
    val_y = one_hot_encode(val_y)

    train = []
    for i in range(train_X.shape[0]):
        labeled_input = list((train_X[i].flatten(), train_y[i]))
        train.append(labeled_input)

    validate = []
    for i in range(val_X.shape[0]):
        labeled_input = list((val_X[i].flatten(), val_y[i]))
        validate.append(labeled_input)

    predict = []
    for i in range(test_X.shape[0]):
        p_input = list((test_X[i].flatten()))
        predict.append(p_input)

    print("Building Neural Network")
    neural_net = MLP_Classifier(3600, 100, 19, iterations = 50, learning_rate = 0.01, momentum = 0.5, rate_decay = 0.0001, output_layer = 'softmax')

    print("Training Neural Network")
    neural_net.fit(train)
    neural_net.test(validate)
    neural_net.predict(predict)


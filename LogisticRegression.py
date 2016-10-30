import numpy
import ProcessImage
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split



if __name__ == "__main__":
    test = False
    test_size = 1000

    train_x = numpy.fromfile('train_x.bin', dtype='uint8')
    train_x = train_x.reshape((100000, 60, 60))

    train_y = numpy.genfromtxt('train_y.csv', delimiter=',', skip_header=1)[:, 1]

    if test:
        train_x = train_x[:test_size]
        train_y = train_y[:test_size]

    print("pre-processing images")
    train_x = ProcessImage.process_images(train_x)
    train_data, validation_data, train_labels, validation_labels = train_test_split(train_x, train_y, test_size=0.2)

    '''
    Uncomment to use logistic regression
    '''
    # classifier = LogisticRegression()
    classifier = SVC()

    print("fitting data")
    classifier.fit(train_data, train_labels)

    '''
    Uncomment to make a prediction
    '''
    # log_reg.predict(validation_data)
    print("scoring data")
    print(classifier.score(validation_data, validation_labels))






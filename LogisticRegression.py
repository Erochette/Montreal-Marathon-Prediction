import numpy
import ProcessImage
import os
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


# preprocesses an image
def preprocess_img(img):
    processed_img = ProcessImage.threshold(img)
    print("p-id:%s processed image" % os.getpid())
    return processed_img.flatten()

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
    pool = Pool()
    train_x = pool.map(preprocess_img, train_x)
    pool.close()

    train_data, validation_data, train_labels, validation_labels = train_test_split(train_x, train_y, test_size=0.2)

    log_reg = LogisticRegression()

    print("fitting data")
    log_reg.fit(train_data, train_labels)

    # log_reg.predict(validation_data)
    print("scoring data")
    print(log_reg.score(validation_data, validation_labels))






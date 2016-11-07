import numpy
import os
from PIL import Image
from scipy.ndimage import *
from multiprocessing import Pool


def load_dataset_cnn(subsetsize=None, train_x_file_name='processed_train_x.bin',
                 test_x_file_name='processed_test_x.bin'):
    #Use different test_x_file_name and train_x_file_name
    subset = subsetsize is None

    train_x = numpy.fromfile(train_x_file_name, dtype='uint8')
    train_x = train_x.reshape((100000, 1, 60, 60))


    test_x = numpy.fromfile(test_x_file_name, dtype='uint8')
    test_x = test_x.reshape((20000, 1, 60, 60))

    train_y = numpy.genfromtxt('train_y.csv', delimiter=',', dtype='int32', skip_header=1)[:, 1]
    if subset:
        train_x = train_x[:subsetsize]
        train_y = train_y[:subsetsize]

    train_x = numpy.array(train_x).astype(numpy.float32)
    train_y = numpy.array(train_y).astype(numpy.int32)
    test_x = numpy.array(test_x).astype(numpy.float32)
    return train_x, train_y, test_x


def process():
    train_x = numpy.fromfile('train_x.bin', dtype='uint8')
    train_x = train_x.reshape((100000, 60, 60))
    test_x = numpy.fromfile('test_x.bin', dtype='uint8')
    test_x = test_x.reshape((20000, 60, 60))

    print("pre-processing images")
    train_x = numpy.array(process_images(train_x))
    test_x = numpy.array(process_images(test_x))

    train_x.astype('uint8').tofile('processed_train_x_small.bin')
    test_x.astype('uint8').tofile('processed_test_x_small.bin')

'''
Process all the images in a numpy array using multiprocessing
'''
def process_images(array_of_images):
    # Opens a pool of processors (pid's)
    pool = Pool()

    # Maps a function on the array of images,
    # namely the preprocess_img function
    array_of_images = pool.map(preprocess_img, array_of_images)

    #closes the pool
    pool.close()
    return array_of_images

'''
preprocesses an image
'''
def preprocess_img(img):
    # Thresholds and Centers the image
    processed_img = center_img(threshold(img))

    #normalize values
    processed_img[processed_img == 255] = 1
    processed_img[processed_img == 0] = 0

    # prints the process id
    # (It's hard to get a counter working so this
    # just lets us know something has been processed)
    print("p-id:%s processed image" % os.getpid())

    # returns a compressed image
    # return zoom(processed_img, 0.5)
    return processed_img

'''
Thresholds an image array to get rid of background noise
'''
def threshold(img_array):
    # Thresholds the values
    img_array[img_array < 250] = 0
    img_array[img_array >= 250] = 255

    # Applies median filters --
    # get's the median value around each pixel
    img_array = median_filter(img_array, size=2)

    # labels the clusters of white pixels
    s = generate_binary_structure(2, 2)
    labeled_array, numpatches = label(img_array, s)

    # Gets the sizes of the labeled clusters
    sizes = sum(img_array, labeled_array, range(1, numpatches+1))
    # Thresholds cluster sizes (only keeps clusters greater than 4000
    large_clusters = numpy.where(sizes > 4000)[0] + 1

    # Applies a mask to only keep the clusters within the map
    max_index = numpy.zeros(numpatches + 1, numpy.uint8)
    # Sets the clusters color to 255 'White'
    max_index[large_clusters] = 255
    img_array = max_index[labeled_array]

    return img_array


'''
Centers the image
Not accurate on images with a divide between the numbers
Will improve later
'''
def center_img(img):
    rows = (img != 0).sum(axis=0)
    cols = (img != 0).sum(axis=1)

    row_rolls = 0
    past_cluster = False
    for x in rows:
        if x != 0:
            past_cluster = True
        if past_cluster:
            if x == 0:
                row_rolls -= 1
        else:
            if x == 0:
                row_rolls += 1

    col_rolls = 0
    past_cluster = False
    for x in cols:
        if x != 0:
            past_cluster = True
        if past_cluster:
            if x == 0:
                col_rolls -= 1
        else:
            if x == 0:
                col_rolls += 1

    c_img = numpy.roll(img, -row_rolls / 2, axis=1)
    c_img = numpy.roll(c_img, -col_rolls / 2, axis=0)
    return c_img


if __name__ == "__main__":
    x = numpy.fromfile('train_x.bin', dtype='uint8')
    x = x.reshape((100000, 60, 60))
    index = 52340
    org = x[index]
    threshold_test = preprocess_img(org.copy())
    print threshold_test.shape
    thresh = Image.fromarray(threshold_test)
    original = Image.fromarray(org)
    thresh.save("threshold_%s.png" % index)
    original.save("original_%s.png" % index)
    thresh.show()
    original.show()

    # Uncomment to process all of the files
    # process()

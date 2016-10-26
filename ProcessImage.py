import numpy
from PIL import Image
from scipy.ndimage import *


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


if __name__ == "__main__":
    x = numpy.fromfile('train_x.bin', dtype='uint8')
    x = x.reshape((100000, 60, 60))
    index = 19
    org = x[index]
    threshold_test = threshold(org.copy())
    thresh = Image.fromarray(threshold_test)
    original = Image.fromarray(org)
    thresh.save("threshold_%s.png" % index)
    original.save("original_%s.png" % index)
    thresh.show()
    original.show()

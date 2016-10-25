import numpy
import copy
from PIL import Image
from scipy.ndimage import *


''' Thresholds an image array to get rid of background noise '''
def threshold(img_array):
    # Thresholds the values, so if the color is less
    # than 250 (extremely light grey) set it to 0 (black).
    no_filter_dict = {}
    for i, arr in enumerate(img_array):
        no_filter_dict[i] = {}
        for j, el in enumerate(arr):
            no_filter_dict[i][j] = False
            if el < 250:
                arr[j] = 0
            else:
                arr[j] = 255

    img_array = median_filter(median_filter(median_filter(img_array, size=1), size=2), size=1)

    for i, arr in enumerate(img_array):
        for j, el in enumerate(arr):
            if el == 255:
                if not no_filter_dict[i][j]:
                    temp_dict = copy.deepcopy(no_filter_dict)
                    temp_dict, area = trace_shape(i, j, img_array, 60, 60, temp_dict, 0)
                    # If the area is too small i.e. there's an outlier white bit, then we can be
                    # confident that that white bit is not really part of the shape.
                    if area < 100:
                        arr[j] = 0
                    else:
                        no_filter_dict.update(temp_dict)

    # Finally do a median filter to set outliers to black,
    # i.e. one white speck by itself should probably be black.
    return img_array


''' Finds the area of a white shape in an image '''
def trace_shape(i, j, array2d, width, height, pt_dict, area):
    pt_dict[i][j] = True
    if i + 1 < width:
        if not pt_dict[i + 1][j]:
            if array2d[i + 1][j] == 255:
                area += 1
                pt_dict, area = trace_shape(i + 1, j, array2d, width, height, pt_dict, area)
        if j - 1 > 0 and not pt_dict[i + 1][j - 1]:
            if array2d[i + 1][j - 1] == 255:
                area += 1
                pt_dict, area = trace_shape(i + 1, j - 1, array2d, width, height, pt_dict, area)
        if j + 1 < height and not pt_dict[i + 1][j + 1]:
            if array2d[i + 1][j + 1] == 255:
                area += 1
                pt_dict, area = trace_shape(i + 1, j + 1, array2d, width, height, pt_dict, area)
    if i - 1 > 0:
        if not pt_dict[i - 1][j]:
            if array2d[i - 1][j] == 255:
                area += 1
                pt_dict, area = trace_shape(i - 1, j, array2d, width, height, pt_dict, area)
        if j - 1 > 0 and not pt_dict[i - 1][j - 1]:
            if array2d[i - 1][j - 1] == 255:
                area += 1
                pt_dict, area = trace_shape(i - 1, j - 1, array2d, width, height, pt_dict, area)
        if j + 1 < height and not pt_dict[i - 1][j + 1]:
            if array2d[i - 1][j + 1] == 255:
                area += 1
                pt_dict, area = trace_shape(i - 1, j + 1, array2d, width, height, pt_dict, area)
    if j - 1 > 0 and not pt_dict[i][j - 1]:
        if array2d[i][j - 1] == 255:
                area += 1
                pt_dict, area = trace_shape(i, j - 1, array2d, width, height, pt_dict, area)
    if j + 1 < height and  not pt_dict[i][j + 1]:
        if array2d[i][j + 1] == 255:
                area += 1
                pt_dict, area = trace_shape(i, j + 1, array2d, width, height, pt_dict, area)

    return pt_dict, area


if __name__ == "__main__":
    x = numpy.fromfile('train_x.bin', dtype='uint8')
    x = x.reshape((100000,60,60))
    index = 0
    org = x[index]
    threshold_test = threshold(org.copy())
    thresh = Image.fromarray(threshold_test) # to visualize only
    original = Image.fromarray(org)
    thresh.save("threshold_%s.png" % index)
    original.save("original_%s.png" % index)
    thresh.show()
    original.show()

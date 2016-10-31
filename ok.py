import numpy
import os
from scipy.ndimage import *
from multiprocessing import Pool

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
    # Thresholds the image
    processed_img = center_img(threshold(img))

    #normalize values
    processed_img[processed_img == 255] = 1.
    processed_img[processed_img == 0] = 0.

    # prints the process id
    # (It's hard to get a counter working so this
    # just lets us know something has been processed)
    print("p-id:%s processed image" % os.getpid())

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
    index = 52363
    org = x[index]
    threshold_test = center_img(threshold(org.copy()))
   # thresh = Image.fromarray(threshold_test)
    #original = Image.fromarray(org)
   # thresh.save("threshold_%s.png" % index)
  #  original.save("original_%s.png" % index)
  #  thresh.show()
#original.show()










#Correctly formating image array
pixels = threshold_test
for y in range(len(pixels)):
    for x in range(len(pixels[y])):
        if pixels[y][x] == 255:
            pixels[y][x] = 1


square_list = [-1,0,1]
neighbor = False
total_weights_bias = []

"""For each pixel, look if there is a classification to do
   If yes, find the weights and bias"""

e = 0
for y in range(len(pixels)):
    for x in range(len(pixels[y])):

        weights_bias = []
        pixel_weight = [0,0]
        pixel_bias = 0.4

        #Look for non-similar neighbor
        for square_y in square_list:
            for square_x in square_list:
                try:
                    if pixels[y+square_y][x+square_x] != pixels[y][x]:
                        neighbor = True
                        break
                except IndexError:
                    continue

        #If there is a non-similar pixel,
        #compute the classification boundary
        if neighbor:

            done = False
            p = 0

            #Loop until no classification error    
            while(not done):

                notDone = True
                p += 1
                if p > 300:
                    e +=1
                    break

                #Loop throuth each neighbor pixel
                for square_y in square_list:
                    for square_x in square_list:

                        #Try to predic classification
                        try:
                            weight_one = pixel_weight[0] * square_list[square_x]
                            weight_two = pixel_weight[1] * square_list[square_y]

                            total = weight_one + weight_two + pixel_bias

                            if total > 0:
                                prediction = 1
                            else:
                                prediction = 0

                            #If bad classification, update weights and bias
                            if prediction == pixels[y+square_y][x+square_x]:
                                continue
                            else:
                                notDone = False
                                pixel_weight[0] = pixel_weight[0] + (pixels[y+square_y][x+square_x]-prediction)*square_list[square_x]
                                pixel_weight[1] = pixel_weight[1] + (pixels[y+square_y][x+square_x]-prediction)*square_list[square_y]
                                pixel_bias = pixel_bias + (pixels[y+square_y][x+square_x]-prediction)

                            done = notDone

                        except IndexError:
                            continue

        #Make a list of all boundaries
            weights_bias.append(pixel_weight[0])
            weights_bias.append(pixel_weight[1])
            weights_bias.append(pixel_bias)
  
        neighbor = False
        total_weights_bias.append(weights_bias)

print total_weights_bias
print e
       
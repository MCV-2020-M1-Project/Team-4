import cv2 as cv
import math
import pickle
import numpy as np
import ml_metrics as metrics
import glob
import time

qsd1file = open('qsd1_w1/gt_corresps.pkl', 'rb')
qsd1 = pickle.load(qsd1file)
bbddfile = open('BBDD/relationships.pkl', 'rb')
bbdd1 = pickle.load(bbddfile)

range_qsd1 = len(qsd1)
range_bbdd = len(bbdd1)

# TASK 1 Create Museum and query image descriptors (BBDD & QS1)
bbdd = []  # Array (np.array) with the histogram of each image in the bbdd folder
qs1 = []  # Array (np.array) with the histogram of each image in the qsd1 folder


def create_hists(image):
    """
    This function load an image, filtered with some basic operations and calculate some specific histograms
    :param image: relative path of image
    :return: array with histograms concatenated
    """
    img = cv.imread(image)
    height, width, dimensions = img.shape
    if height > 250:
        factor = height//250
        img = cv.resize(img, (width//factor, height//factor), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    height, width, dimensions = img.shape

    # Number of divisions
    column = 4
    row = 16
    height_cutoff = height // row
    width_cutoff = width // column
    output_array = []

    for d in range(dimensions):
        # Adaptable number of bins, to give each dimension more or less weight in the final evaluation
        if d == 0:
            bins = 16
        else:
            bins = 32
        for c in range(column):
            for r in range(row):
                s1 = img[r * height_cutoff:(r + 1) * height_cutoff, c * width_cutoff: (c + 1) * width_cutoff, d]
                s1_hist = np.array(cv.calcHist([s1], [0], None, [bins], [0, 256]))
                cv.normalize(s1_hist, s1_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
                output_array = np.concatenate((output_array, s1_hist), axis=None)
    return output_array


# Old function, is currently used because there is some issues with the general load function "load_img_folder"
def qs_load():
    """
    This function create ddbb.txt file using all the image from qsd1_w1 with jpg format
    :return: Nothing
    """
    with open('ddbb.txt', 'w') as out_file:
        for i in range(range_qsd1):
            if i < 10:
                image = 'qsd1_w1/0000' + str(i) + '.jpg'
            else:
                image = 'qsd1_w1/000' + str(i) + '.jpg'
            his = create_hists(image)
            qs1.append(his)
            for v in range(len(his)):
                out_file.write(str(his[v]))
                if v < len(his) - 1:
                    out_file.write('\t')
            out_file.write('\n')


# Old function, is currently used because there is some issues with the general load function "load_img_folder"
def bbdd_load():
    """
    This function load each image from BBDD folder listed in relationship.pkl with jpg format
    :return: Nothing
    """
    for i in range(range_bbdd):
        if i < 10:
            image = 'BBDD/bbdd_0000' + str(i) + '.jpg'
        elif i < 100:
            image = 'BBDD/bbdd_000' + str(i) + '.jpg'
        else:
            image = 'BBDD/bbdd_00' + str(i) + '.jpg'
        bbdd.append(create_hists(image))


def load_img_folder(path, file_path_output):
    """
    General automated load image function, all the jpg files in the folder "path" will be evaluated using create_hist() and
    saved the results in "file_path_output"
    :param path: Folder where input image are stored
    :param file_path_output: file path + file name where will be stored image descriptors
    :return: Nothing
    """
    with open(file_path_output, 'w') as out_file:
        for image in glob.glob(path + '*.jpg'):
            his = create_hists(image)
            for v in range(len(his)):
                out_file.write(str(his[v]))
                if v < len(his) - 1:
                    out_file.write('\t')
            out_file.write('\n')


def load_porcessed(path):
    """
    General automated load image descriptors function, it does not work in linux and macOS
    :param path: determinate the path + name of the descriptor file
    :return: descriptors
    """
    lines = np.loadtxt(path, delimiter='\t')
    return lines


# TASK 2 Implement / compute similarity measures to compare images

def euclidean(h1, h2):  # Euclidean distance
    """
    Evaluate h1 and h2 with euclidean relation
    :param h1: histogram of looking image
    :param h2: histogram data set
    :return: distance for each histogram bin
    """
    result = 0
    for k in range(len(h1)):
        dif = (h1[k] - h2[k]) ** 2
        result += dif

    return math.sqrt(result)


def l1distance(h1, h2):  # L1 distance
    """
    Evaluate h1 and h2 with lineal distance
    :param h1: histogram of looking image
    :param h2: histogram of data set
    :return: distance for each histogram bin
    """
    result = 0
    for k in range(len(h1)):
        dif = abs(h1[k] - h2[k])
        result += dif
    return result


def x2distance(h1, h2):  # x^2 distance
    """
    Evaluate h1 and h2 with quadratic distance
    :param h1: histogram of looking image
    :param h2: histogram of data set
    :return: result for each histogram bin
    """
    result = 0
    l = len(h1)
    for k in range(l):
        # This if is a short-cut, after evaluate half image it check if the value is too high (threshold found it
        # empirically)
        if k == l/2:
            if (l*0.11)/2 < result:
                result = l
                return result

        h11 = h1[k]
        h22 = h2[k]
        if (h11 + h22) == 0:
            dif = 0
        else:
            dif = ((h11 - h22) ** 2) / (h11 + h22)

        result += dif

    return result


def histogram_sequence():
    print("Starting histogram sequence...")
    t = time.time()

    # Creating variables
    result_1k = []
    result_5k = []
    min_val = 0

    # Comment it if you already have folders created
    print("Creating image descriptors...")
    load_img_folder('qsd1_w1/', 'qs1.txt')
    load_img_folder('BBDD/', 'bbdd.txt')
    print("Created")
    # ---------------------------------
    print("Loading descriptors...")
    qs1 = load_porcessed('qs1.txt')
    bbdd = load_porcessed('bbdd.txt')
    print("Loaded")

    # Compare each picture  of both folders
    print("Finding similarities")
    for i in range(range_qsd1):
        h1 = qs1[i]
        distance = {}

        for key in range(range_bbdd):
            distance[key] = x2distance(h1, bbdd[key])
            min_val = min(distance.values())

        # Sort 5 best results
        x = sorted(distance, key=distance.get, reverse=False)[:5]
        result_5k.append(x)
        # Push the position of minimum value
        result = [key for key, value in distance.items() if value == min_val]
        result_1k.append(result)

    # TASK 4 Evaluation using map@k
    score_k1 = metrics.mapk(qsd1, result_1k, 1) * 100
    score_k5 = metrics.mapk(qsd1, result_5k, 5) * 100

    print('Score K1 = ', score_k1, '%')
    print('Score K5 = ', score_k5, '%')

    # Print execution time, just for extra information
    t = time.time()-t
    print("time needed to complete sequence: ", t)
    print("for each image (aprox): ", t / range_qsd1)


if __name__ == "__main__":
    histogram_sequence()

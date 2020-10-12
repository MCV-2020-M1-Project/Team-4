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
    imgRaw = cv.imread(image)
    height, width, dimensions = imgRaw.shape
    if height > 250:
        factor = height//250
        imgRaw = cv.resize(imgRaw, (width//factor, height//factor), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(imgRaw, cv.COLOR_BGR2GRAY)
    imgRaw = cv.cvtColor(imgRaw, cv.COLOR_BGR2Lab)
    height, width = img.shape
    # Cut the image in 4
    column = 4
    row = 16
    height_cutoff = height // row
    width_cutoff = width // column
    outputArray = []
    #for c in range(column):
    #    for r in range(row):
    #        s1 = img[r * height_cutoff:(r + 1) * height_cutoff, c * width_cutoff: (c + 1) * width_cutoff]
    #        s1_hist = np.array(cv.calcHist([s1], [0], None, [32], [0, 256]))
    #        cv.normalize(s1_hist, s1_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    #        outputArray = np.concatenate((outputArray, s1_hist), axis=

    dimensions = 3
    for d in range(dimensions):
        for c in range(column):
            for r in range(row):
                s1 = imgRaw[r * height_cutoff:(r + 1) * height_cutoff, c * width_cutoff: (c + 1) * width_cutoff, d]
                s1_hist = np.array(cv.calcHist([s1], [0], None, [32], [0, 256]))
                cv.normalize(s1_hist, s1_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
                outputArray = np.concatenate((outputArray, s1_hist), axis=None)
    return outputArray


def qs_load():
    with open('ddbb.txt', 'w') as out_file:
        for i in range(range_qsd1):
            if i < 10:
                image = 'qsd1_w1/0000' + str(i) + '.jpg'
            else:
                image = 'qsd1_w1/000' + str(i) + '.jpg'
            # qs1[i] = create_hists(image)
            his = create_hists(image)
            qs1.append(his)
            for v in range(len(his)):
                out_file.write(str(his[v]))
                if v < len(his) - 1:
                    out_file.write('\t')
            out_file.write('\n')


# Old funtion,
def bbdd_load():
    for i in range(range_bbdd):
        if i < 10:
            image = 'BBDD/bbdd_0000' + str(i) + '.jpg'
        elif i < 100:
            image = 'BBDD/bbdd_000' + str(i) + '.jpg'
        else:
            image = 'BBDD/bbdd_00' + str(i) + '.jpg'
        bbdd.append(create_hists(image))


def load_img_folder(path, file_path_output):
    with open(file_path_output, 'w') as out_file:
        for image in glob.glob(path + '*.jpg'):
            his = create_hists(image)
            for v in range(len(his)):
                out_file.write(str(his[v]))
                if v < len(his) - 1:
                    out_file.write('\t')
            out_file.write('\n')


def load_porcessed(path):
    lines = np.genfromtxt(path, delimiter='\t')
    return lines


# TASK 2 Implement / compute similarity measures to compare images


def euclidean(h1, h2):  # Euclidean distance
    result = 0
    for k in range(len(h1)):
        dif = (h1[k] - h2[k]) ** 2
        result += dif

    return math.sqrt(result)


def l1distance(h1, h2):  # L1 distance
    result = 0
    for k in range(len(h1)):
        dif = abs(h1[k] - h2[k])
        result += dif
    return result


def x2distance(h1, h2):  # x^2 distance
    result = 0
    l= len(h1)
    for k in range(l):
        if k == l/2:
            if (l*0.12)/2 < result:
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
    t = time.time()
    # Finding Euclidean distance
    print("Starting histogram sequence...")
    result_1k = []
    result_5k = []
    min_val = 0
    # Comment if you already have folders created
    print("Creating image descriptors...")
    load_img_folder('qsd1_w1/', 'qs1.txt')
    load_img_folder('BBDD/', 'bbdd.txt')
    print("Created")
    # ---------------------------------
    print("Loading descriptors...")
    qs1 = load_porcessed('qs1.txt')
    bbdd = load_porcessed('bbdd.txt')
    print("Loaded")
    print("Finding similarities")
    for i in range(range_qsd1):
        h1 = qs1[i]
        distance = {}
        for key in range(range_bbdd):
            distance[key] = x2distance(h1, bbdd[key])
            #print(distance[key])
            min_val = min(distance.values())
        x = sorted(distance, key=distance.get, reverse=False)[:5]
        result_5k.append(x)
        result = [key for key, value in distance.items() if value == min_val]
        result_1k.append(result)
        # print('The image that corresponds to the query image nº ', i, ' is ', qsd1[i])
        # print('The image with the lowest euclidean distance is ', result)

    score_k1 = metrics.mapk(qsd1, result_1k, 1) * 100
    score_k5 = metrics.mapk(qsd1, result_5k, 5) * 100

    print('Score K1 = ', score_k1, '%')
    print('Score K5 = ', score_k5, '%')
    t = time.time()-t
    print("time needed to complete sequence: ", t)
    print("for each image (aprox): ", t / range_qsd1)

if __name__ == "__main__":
    histogram_sequence()

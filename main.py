import cv2 as cv
import math
import pickle
import numpy as np
import ml_metrics as metrics

qsd1file = open('qsd1_w1/gt_corresps.pkl', 'rb')
qsd1 = pickle.load(qsd1file)
bbddfile = open('BBDD/relationships.pkl', 'rb')
bbdd1 = pickle.load(bbddfile)

range_qsd1 = len(qsd1)
range_bbdd = len(bbdd1)

# TASK 1 Create Museum and query image descriptors (BBDD & QS1)
bbdd = {}  # Dictionary with the histogram of each image in the bbdd folder
qs1 = {}  # Dictionary with the histogram of each image in the qsd1 folder


def create_hists(image):
    img = cv.imread(image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width = img.shape
    # Cut the image in 4
    height_cutoff = height // 4
    # width_cutoff = width // 4
    s1 = img[:height_cutoff, :]
    s2 = img[height_cutoff:height_cutoff * 2, :]
    s3 = img[height_cutoff * 2:height_cutoff * 3, :]
    s4 = img[height_cutoff * 3:, :]
    s1_hist = np.array(cv.calcHist([s1], [0], None, [256], [0, 256]))
    s2_hist = np.array(cv.calcHist([s2], [0], None, [256], [0, 256]))
    s3_hist = np.array(cv.calcHist([s3], [0], None, [256], [0, 256]))
    s4_hist = np.array(cv.calcHist([s4], [0], None, [256], [0, 256]))
    cv.normalize(s1_hist, s1_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    cv.normalize(s2_hist, s2_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    cv.normalize(s3_hist, s3_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    cv.normalize(s4_hist, s4_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    return np.concatenate((s1_hist, s2_hist, s3_hist, s4_hist), axis=None)


for i in range(range_qsd1):
    if i < 10:
        image = 'qsd1_w1/0000' + str(i) + '.jpg'
    else:
        image = 'qsd1_w1/000' + str(i) + '.jpg'
    qs1[i] = create_hists(image)

for i in range(range_bbdd):
    if i < 10:
        image = 'BBDD/bbdd_0000' + str(i) + '.jpg'
    elif i < 100:
        image = 'BBDD/bbdd_000' + str(i) + '.jpg'
    else:
        image = 'BBDD/bbdd_00' + str(i) + '.jpg'
    bbdd[i] = create_hists(image)


# TASK 2 Implement / compute similarity measures to compare images


def euclidean(h1, h2):  # Euclidean distance
    sum = 0
    for k in range(256 * 4):
        dif = (h1[k] - h2[k]) ** 2
        sum += dif

    return math.sqrt(sum)


def l1distance(h1, h2):  # L1 distance
    sum = 0
    for k in range(256*4):
        dif = abs(h1[k] - h2[k])
        sum += dif
    return sum


def x2distance(h1, h2):  # x^2 distance
    sum = 0
    for k in range(256*4):
        if (h1[k] + h2[k]) == 0:
            dif = 0
        else:
            dif = ((h1[k] - h2[k])**2)/(h1[k] + h2[k])
        sum += dif

    return sum


# Finding Euclidean distance
result_1k = []
result_5k = []
for i in range(range_qsd1):
    h1 = qs1[i]
    distance = {}
    for key in bbdd:
        distance[key] = x2distance(h1, bbdd[key])
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

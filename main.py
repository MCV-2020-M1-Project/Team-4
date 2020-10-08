import cv2 as cv
import math
import pickle

qsd1file = open('qsd1_w1/gt_corresps.pkl', 'rb')
qsd1 = pickle.load(qsd1file)
bbddfile = open('BBDD/relationships.pkl', 'rb')
bbdd1 = pickle.load(bbddfile)

range_qsd1=len(qsd1)
range_bbdd=len(bbdd1)

# TASK 1 Create Museum and query image descriptors (BBDD & QS1)
bbdd = {}  # Dictionary with the histogram of each image in the bbdd folder
qs1 = {}  # Dictionary with the histogram of each image in the qsd1 folder
for i in range(range_qsd1):
    if i < 10:
        image = '0000' + str(i) + '.jpg'
    else:
        image = '000' + str(i) + '.jpg'

    img = cv.imread('qsd1_w1/' + image, 0)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    qs1[i] = hist

for i in range(range_bbdd):
    if i < 10:
        image = 'bbdd_0000' + str(i) + '.jpg'
    elif i < 100:
        image = 'bbdd_000' + str(i) + '.jpg'
    else:
        image = 'bbdd_00' + str(i) + '.jpg'

    img = cv.imread('BBDD/' + image, 0)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    bbdd[i] = hist


# TASK 2 Implement / compute similarity measures to compare images

def euclidean(h1, h2):  # Euclidean distance
    sum = 0
    for k in range(256):
        dif = (h1[k] - h2[k]) ** 2
        sum += dif

    return math.sqrt(sum)


def l1distance(h1, h2):  # L1 distance
    sum = 0
    for k in range(256):
        dif = abs(h1[k] - h2[k])
        sum += dif

    return sum


def x2distance(h1, h2):  # x^2 distance
    for k in range(256):
        dif = ((h1[k] - h2[k])**2)/(h1[k] + h2[k])
        sum += dif

    return sum


# Finding x^2 distance in the image 0
h1 = qs1[0]
distance = {}
for key in bbdd:
    distance[key] = x2distance(h1, bbdd[key])
min_val = min(distance.values())

result = [key for key, value in distance.items() if value == min_val]

print('The image that corresponds to the first query image is ', qsd1[0])
print('The image with lower euclidean distance is ', result)

import cv2
import imutils
import math
import pickle
import numpy as np
import ml_metrics as metrics
import time
from docopt import docopt
import statistics as stats
from background import BackgroundRemove


def create_hists(image):
    if count == 0:
        image = cv2.imread(image)
    height, width, dimensions = image.shape
    if height > 250:
        factor = height//250
        image = cv2.resize(image, (width//factor, height//factor), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    height, width = img.shape
    # Cut the image in 4
    column = 4
    row = 16
    height_cutoff = height // row
    width_cutoff = width // column
    outputArray = []

    dimensions = 3
    for d in range(dimensions):
        for c in range(column):
            for r in range(row):
                s1 = image[r * height_cutoff:(r + 1) * height_cutoff, c * width_cutoff: (c + 1) * width_cutoff, d]
                s1_hist = np.array(cv2.calcHist([s1], [0], None, [32], [0, 256]))
                cv2.normalize(s1_hist, s1_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                outputArray = np.concatenate((outputArray, s1_hist), axis=None)
    return outputArray

def rgb_hist_3d(image, bins=8, mask=None):
    
    if count == 0:
        image = cv2.imread(image)
    hist = cv2.calcHist([image], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()


def openfile():
    
    qsd1file = open('qsd1_w1/gt_corresps.pkl', 'rb')
    qsd1 = pickle.load(qsd1file)
    bbddfile = open('BBDD/relationships.pkl', 'rb')
    bbdd1 = pickle.load(bbddfile)
    qsd2file = open('qsd2_w1/gt_corresps.pkl', 'rb')
    qsd2 = pickle.load(qsd2file)
    mask2file = open('qsd2_w1/frames.pkl', 'rb')
    mask2 = pickle.load(mask2file)
    
    return qsd1, bbdd1, qsd2, mask2


    
# TASK 2 Implement / compute similarity measures to compare images
def euclidean(h1, h2):  # Euclidean distance
    result = 0
    for k in range(len(h1)):
        dif = (h1[k] - h2[k]) ** 2
        result += dif

    return math.sqrt(result)


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
    
if __name__ == "__main__":
    
    qsd1, bbdd1, qsd2, mask2 = openfile()
    range_qsd1 = len(qsd1)
    range_qsd2 = len(qsd2)
    range_mask2 = len(mask2)
    range_bbdd1 = len(bbdd1)
    
    qs1 = []  # Array (np.array) with the histogram of each image in the qsd1 folder
    qs2 = []  # Array (np.array) with the histogram of each image in the qsd1 folder
    bbdd = []  # Array (np.array) with the histogram of each image in the bbdd folder 
    qs2 = []  # Array (np.array) with the histogram of each image in the qsd1 folder
    maskbis = [] # Array (np.array) with the histogram of each image in the mask folder
    count=0
    
    
    t = time.time()
    # Finding Euclidean distance
    print("Starting histogram sequence...")
    result_1k = []
    result_5k = []
    result_10k = []
    min_val = 0
    print("Creating image descriptors...")
    
    
    for i in range(range_qsd1):
        if i < 10:
            image = 'qsd1_w1/0000' + str(i) + '.jpg'
        else:
            image = 'qsd1_w1/000' + str(i) + '.jpg'
    

        qs1.append(create_hists(image))
        
    
    for i in range(range_qsd2):
        if i < 10:
            image = 'qsd2_w1/0000' + str(i) + '.jpg'
        else:
            image = 'qsd2_w1/000' + str(i) + '.jpg'
    
        count=1
        qs = cv2.imread(image)
        qs, mask = BackgroundRemove.remove_background(qs,2)
        maskbis.append(mask)
        qs2.append(create_hists(qs))
        count=0

    for i in range(range_bbdd1):
        if i < 10:
            image = 'BBDD/bbdd_0000' + str(i) + '.jpg'
        elif i < 100:
            image = 'BBDD/bbdd_000' + str(i) + '.jpg'
        else:
            image = 'BBDD/bbdd_00' + str(i) + '.jpg'
            

        bbdd.append(create_hists(image))

    
    for i in range(range_mask2):
        if i < 10:
            image = 'qsd2_w1/0000' + str(i) + '.png'


        elif i < 100:
            image = 'qsd2_w1/000' + str(i) + '.png'
           
        mask = cv2.imread(image)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask2.append(mask)
    mask2 = mask2[30:]
        
    print("Loaded")
    print("Finding similarities")

    for i in range(range_qsd2):
        h1 = qs2[i]
        distance = {}
        for key in range(range_bbdd1):
            distance[key] = euclidean(h1, bbdd[key])
            #print(distance[key])
            min_val = min(distance.values())
        x = sorted(distance, key=distance.get, reverse=False)[:5]
        result_5k.append(x)
        result = [key for key, value in distance.items() if value == min_val]
        y = sorted(distance, key=distance.get, reverse=False)[:10]
        result_10k.append(y)
        result = [key for key, value in distance.items() if value == min_val]
        result_1k.append(result)
        # print('The image that corresponds to the query image nº ', i, ' is ', qsd1[i])
        # print('The image with the lowest euclidean distance is ', result)

    score_k1 = metrics.mapk(qsd2, result_1k, 1) * 100
    score_k5 = metrics.mapk(qsd2, result_5k, 5) * 100
    score_k10 = metrics.mapk(qsd2, result_10k, 10) * 100

    print('Score K1 = ', score_k1, '%')
    print('Score K5 = ', score_k5, '%')
    print('Score K10 = ', score_k10, '%')
    t = time.time()-t
    print("time needed to complete sequence: ", t)
    print("for each image (aprox): ", t / range_qsd1)


"""
Arreglar!!!!


precision = []
recall = []
score_f1 = []
for i in range(range_qsd2):
    pre, rec, f1 = BackgroundRemove.evaluate_mask(mask2[i], maskbis[i])
    precision.append(pre)
    recall.append(rec)
    score_f1.append(f1)
    


a = stats.median(precision)
b = stats.median(recall)
c = stats.median(score_f1)

"""

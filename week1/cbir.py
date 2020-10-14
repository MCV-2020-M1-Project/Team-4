"""

Usage:
  cbir_def.py <weekNumber> <teamNumber> <winEval> <querySet> <MethodNumber> [--testDir=<td>] 
  cbir_def.py -h | --help
Options:
  --testDir=<td>        Directory with the test images & masks [default: /home/dlcv/DataSet/fake_test]        ###Aixo del dir no ho tinc clar###
  
"""


import os
import sys
import cv2 as cv
import math
import pickle
import numpy as np
import ml_metrics as metrics
import glob
import time
import pandas as pd
from docopt import docopt


def openfile():
    
    qsd1file = open('qsd{}_w{}/gt_corresps.pkl'.format(query_set, week), 'rb')
    qsd1 = pickle.load(qsd1file)
    bbddfile = open('BBDD/relationships.pkl', 'rb')
    bbdd1 = pickle.load(bbddfile)
    
    return qsd1, bbdd1


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


def rgb_hist_3d(image, bins=8, mask=None):
    
    imgRaw = cv.imread(image)
    hist = cv.calcHist([imgRaw], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist)
    return hist.flatten()

# Old funtion,
def bbdd_load():
    bbdd = []
    for i in range(range_bbdd):
        if i < 10:
            image = 'BBDD/bbdd_0000' + str(i) + '.jpg'
        elif i < 100:
            image = 'BBDD/bbdd_000' + str(i) + '.jpg'
        else:
            image = 'BBDD/bbdd_00' + str(i) + '.jpg'
        bbdd.append(create_hists(image))
            
    return bbdd

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


def histogram_sequence():
    
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
    
        if method == 1:
            qs1.append(create_hists(image))
        elif method == 2:
            qs1.append(rgb_hist_3d(image))

    for i in range(range_bbdd):
        if i < 10:
            image = 'BBDD/bbdd_0000' + str(i) + '.jpg'
        elif i < 100:
            image = 'BBDD/bbdd_000' + str(i) + '.jpg'
        else:
            image = 'BBDD/bbdd_00' + str(i) + '.jpg'
            
        if method == 1:
            bbdd.append(create_hists(image))
        elif method == 2:
            bbdd.append(rgb_hist_3d(image))
        
        
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
        y = sorted(distance, key=distance.get, reverse=False)[:10]
        result_10k.append(y)
        result = [key for key, value in distance.items() if value == min_val]
        result_1k.append(result)
        # print('The image that corresponds to the query image nº ', i, ' is ', qsd1[i])
        # print('The image with the lowest euclidean distance is ', result)

    score_k1 = metrics.mapk(qsd1, result_1k, 1) * 100
    score_k5 = metrics.mapk(qsd1, result_5k, 5) * 100
    score_k10 = metrics.mapk(qsd1, result_10k, 10) * 100

    print('Score K1 = ', score_k1, '%')
    print('Score K5 = ', score_k5, '%')
    print('Score K10 = ', score_k10, '%')
    t = time.time()-t
    print("time needed to complete sequence: ", t)
    print("for each image (aprox): ", t / range_qsd1)
    

if __name__ == "__main__":

    # read arguments
    args = docopt(__doc__)

    week      = int(args['<weekNumber>']) #1
    team      = int(args['<teamNumber>']) #04
    query_set = int(args['<querySet>']) #1 or 2
    method = int(args['<MethodNumber>']) #1: divided_hist  2:rgb_3d
    
    # This folder contains your results: mask imaged and window list pkl files. Do not change this.
    results_dir = '/home/dlcv{:02d}/m1-results/week{}/QST{}'.format(team, week, query_set)
    
    qsd1, bbdd1 = openfile()
    range_qsd1 = len(qsd1)
    range_bbdd = len(bbdd1)

    # TASK 1 Create Museum and query image descriptors (BBDD & QS1)
    bbdd = []  # Array (np.array) with the histogram of each image in the bbdd folder 
    qs1 = []  # Array (np.array) with the histogram of each image in the qsd1 folder

    a = histogram_sequence()
    
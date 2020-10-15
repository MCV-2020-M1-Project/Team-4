"""

Usage:
  cbir.py <weekNumber> <teamNumber> <winEval> <querySet> <MethodNumber> [--testDir=<td>] 
  cbir.py -h | --help
Options:
  --testDir=<td>        Directory with the test images & masks [default: /home/dlcv/DataSet/fake_test]        ###Aixo del dir no ho tinc clar###
  
"""

import pickle
import time
import cv2

import ml_metrics as metrics
from docopt import docopt

from query_images import HistogramGenerator, HistogramDistance


def openfile():
    
    qsd1file = open('qsd{}_w{}/gt_corresps.pkl'.format(query_set, week), 'rb')
    qsd1 = pickle.load(qsd1file)
    bbddfile = open('BBDD/relationships.pkl', 'rb')
    bbdd1 = pickle.load(bbddfile)
    
    return qsd1, bbdd1


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
        image = cv2.imread('qsd{}_w{}/{:05d}.jpg'.format(query_set, week, i))
    
        if method == 1:
            qs1.append(HistogramGenerator.create_hists(image))
        elif method == 2:
            qs1.append(HistogramGenerator.rgb_hist_3d(image))

    for i in range(range_bbdd):
        image = cv2.imread('BBDD/bbdd_{:05d}.jpg'.format(i))
            
        if method == 1:
            bbdd.append(HistogramGenerator.create_hists(image))
        elif method == 2:
            bbdd.append(HistogramGenerator.rgb_hist_3d(image))
        
        
    print("Loaded")
    print("Finding similarities")

    for i in range(range_qsd1):
        h1 = qs1[i]
        distance = {}
        for key in range(range_bbdd):
            distance[key] = HistogramDistance.x2distance(h1, bbdd[key])
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

    histogram_sequence()

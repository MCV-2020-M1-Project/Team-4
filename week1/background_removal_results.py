"""

Usage:
  background_removal_results.py <weekNumber> <teamNumber> <winEval> <querySet> <MethodNumber> <distanceMeasure> [--testDir=<td>] 
  background_removal_results.py -h | --help
  
  <weekNumber> --> Number of the week
  <teamNumber> --> Team Number, in our case 04
  <winEval> --> 0 for the first week and 1 for the rest of weeks
  <querySet> --> number of the query
  <MethodNumber> --> Number of the method : 1: Edges, 2: Morph
  <distanceMeasure> --> 1: Euclidean distance, 2: x^2 distance
  
  Example of use --> python background_removal_results.py 1 04 0 2 2 1
          
"""

import pickle
import time

import cv2
import ml_metrics as metrics
from docopt import docopt
import statistics as stats

from query_images import BackgroundRemove, HistogramDistance, HistogramGenerator


def openfile():
    qsd1file = open('qsd1_w1/gt_corresps.pkl', 'rb')
    qsd1 = pickle.load(qsd1file)
    bbddfile = open('BBDD/relationships.pkl', 'rb')
    bbdd1 = pickle.load(bbddfile)
    qsd2file = open('qsd{}_w{}/gt_corresps.pkl'.format(query_set, week), 'rb')
    qsd2 = pickle.load(qsd2file)
    mask2file = open('qsd{}_w{}/frames.pkl'.format(query_set, week), 'rb')
    mask2 = pickle.load(mask2file)

    return qsd1, bbdd1, qsd2, mask2

def stadistics_mask():
    precision = []
    recall = []
    score_f1 = []
    for i in range(range_qsd2):
        pre, rec, f1 = BackgroundRemove.evaluate_mask(mask2[i], maskbis[i])
        precision.append(pre)
        recall.append(rec)
        score_f1.append(f1)
    
    return precision, recall, score_f1


if __name__ == "__main__":
    
    
    # read arguments
    args = docopt(__doc__)

    week      = int(args['<weekNumber>']) #1
    team      = int(args['<teamNumber>']) #04
    query_set = int(args['<querySet>']) #1 or 2
    method = int(args['<MethodNumber>']) #1: divided_hist  2:rgb_3d
    distance_m = int(args['<distanceMeasure>']) # 1: euclidean and 2: x^2 distance
    
    # This folder contains your results: mask imaged and window list pkl files. Do not change this.
    results_dir = '/Users/danielyuste/Documents/Master/M1_Project/week1'
    

    qsd1, bbdd1, qsd2, mask2 = openfile()
    range_qsd1 = len(qsd1)
    range_qsd2 = len(qsd2)
    range_mask2 = len(mask2)
    range_bbdd1 = len(bbdd1)

    qs1 = []  # Array (np.array) with the histogram of each image in the qsd1 folder
    qs2 = []  # Array (np.array) with the histogram of each image in the qsd1 folder
    bbdd = []  # Array (np.array) with the histogram of each image in the bbdd folder 
    qs2 = []  # Array (np.array) with the histogram of each image in the qsd1 folder
    maskbis = []  # Array (np.array) with the histogram of each image in the mask folder
    count = 0

    t = time.time()
    # Finding Euclidean distance
    print("Starting histogram sequence...")
    result_1k = []
    result_5k = []
    result_10k = []
    min_val = 0
    print("Creating image descriptors...")

    for i in range(range_qsd1):
        img = cv2.imread('qsd1_w1/{:05d}.jpg'.format(i))
        qs1.append(HistogramGenerator.create_hists(img))

    for i in range(range_qsd2):
        image = 'qsd{}_w{}/{:05d}.jpg'.format(query_set, week, i)

        count = 1
        qs = cv2.imread(image)
        qs, mask = BackgroundRemove.remove_background(qs, method)
        maskbis.append(mask)
        qs2.append(HistogramGenerator.create_hists(qs))
        count = 0

    for i in range(range_bbdd1):
        img = cv2.imread('BBDD/bbdd_{:05d}.jpg'.format(i))
        bbdd.append(HistogramGenerator.create_hists(img))

    for i in range(range_mask2):
        image = 'qsd{}_w{}/{:05d}.png'.format(query_set, week, i)

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
            if distance_m ==2:
                distance[key] = HistogramDistance.x2distance(h1, bbdd[key])
                min_val = min(distance.values())
            elif distance_m ==1:
                distance[key] = HistogramDistance.euclidean(h1, bbdd[key])
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

    precision, recall, score_f1 = stadistics_mask()
    
    print('Precision= ',stats.median(precision)*100 , '%')
    print('Recall = ', stats.median(recall)*100, '%')
    print('Score_f1 = ', stats.median(score_f1)*100, '%')
    
    t = time.time() - t
    print("time needed to complete sequence: ", t)
    print("for each image (aprox): ", t / range_qsd1)
    

    
    #Write the results in a .pkl file
    pickle_file = '{}/query{}/method{}/result.pkl'.format(results_dir, query_set, method)
    f = open(pickle_file, 'wb')
    pickle.dump((qsd2, result_10k), f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close

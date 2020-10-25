"""

Usage:
  cbir.py <weekNumber> <teamNumber> <winEval> <querySet> <MethodNumber> <distanceMeasure> [--testDir=<td>] 
  cbir.py -h | --help
  
  <weekNumber> --> Number of the week
  <teamNumber> --> Team Number, in our case 04
  <winEval> --> 0 for the first week and 1 for the rest of weeks
  <querySet> --> number of the query
  <MethodNumber> --> Number of the method : 1: Divided Histogram, 2: 3d Color Histogram
  <distanceMeasure> --> 1: Euclidean distance, 2: x^2 distance
  
  Example of use --> python cbir.py 1 04 0 1 1 2

"""

import pickle
import time
import cv2

import ml_metrics as metrics
from docopt import docopt


from query_images import HistogramGenerator, HistogramDistance, TextDetection



def openfile():
    
    qsd1file = open('qsd{}_w{}/gt_corresps.pkl'.format(query_set, week), 'rb')
    qsd1 = pickle.load(qsd1file)
    bbddfile = open('BBDD/relationships.pkl', 'rb')
    bbdd1 = pickle.load(bbddfile)
    
    return qsd1, bbdd1


def histogram_sequence(row, column):
    
    t = time.time()
    # Finding Euclidean distance
    print("Starting histogram sequence...")
    result_1k = []
    result_5k = []
    result_10k = []
    min_val = 0
    print("Creating image descriptors...")
    
    for i in range(range_qsd1):
        
        
        if week ==1:
            image = cv2.imread('qsd{}_w{}/{:05d}.jpg'.format(query_set, week, i))
            if method == 1:
                qs1.append(HistogramGenerator.create_hists(image, row, column))
            elif method == 2:
                qs1.append(HistogramGenerator.rgb_hist_3d(image))

            for i in range(range_bbdd):
                image = cv2.imread('BBDD/bbdd_{:05d}.jpg'.format(i))
            
                if method == 1:
                    bbdd.append(HistogramGenerator.create_hists(image, row, column))
                elif method == 2:
                    bbdd.append(HistogramGenerator.rgb_hist_3d(image))
                    
        elif week == 2:
            #Using always the method one
            for i in range(range_qsd1):
                image = cv2.imread('qsd{}_w{}/{:05d}.jpg'.format(query_set, week, i))
                #Extract text coordinates and remove it on the image
                coordinates, mask = TextDetection.text_detection(image)
                qs1.append(HistogramGenerator.create_hists(image, row, column, mask))

                
                
    for i in range(range_bbdd):
        image = cv2.imread('BBDD/bbdd_{:05d}.jpg'.format(i))
        #image[int(coordinates[1]-5):int(coordinates[3]+5), int(coordinates[0]-5):int(coordinates[2]+5)]= 0
        bbdd.append(HistogramGenerator.create_hists(image, row, column))

            
        
    print("Loaded")
    print("Finding similarities")

    for i in range(range_qsd1):
        h1 = qs1[i]
        distance = {}
        for key in range(range_bbdd):
            if week == 1: 
                if distance_m == 2:
                    distance[key] = HistogramDistance.x2distance(h1, bbdd[key])
                    min_val = min(distance.values())
                elif distance_m == 1:
                    distance[key] = HistogramDistance.euclidean(h1, bbdd[key])
                    min_val = min(distance.values())
            if week ==2:
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
    
    metric_iou_mean, query_list, tboxes_list = TextDetection.extract_text()
    
    return result_10k, metric_iou_mean, query_list, tboxes_list


    

if __name__ == "__main__":

    # read arguments
    args = docopt(__doc__)

    week      = int(args['<weekNumber>']) #1
    team      = int(args['<teamNumber>']) #04
    query_set = int(args['<querySet>']) #1 or 2
    method = int(args['<MethodNumber>']) #1: divided_hist  2:rgb_3d
    distance_m = int(args['<distanceMeasure>']) # 1: euclidean and 2: x^2 distance
    
    # This folder contains your results: mask imaged and window list pkl files. Do not change this.
    #results_dir = '/home/dlcv{:02d}/m1-results/week{}/QST{}/Method{}'.format(team, week, query_set, method)
    results_dir = '/Users/danielyuste/Documents/Master/M1_Project/week2a'
    #Select the level of the histogram block
    
    row = input('Please, choose the number of rows to make the histogram block')
    row = int(row)
    column = input('Please, choose the number of columns to make the histogram block')
    column = int(column)
    
    qsd1, bbdd1 = openfile()
    range_qsd1 = len(qsd1)
    range_bbdd = len(bbdd1)

    # TASK 1 Create Museum and query image descriptors (BBDD & QS1)
    bbdd = []  # Array (np.array) with the histogram of each image in the bbdd folder 
    qs1 = []  # Array (np.array) with the histogram of each image in the qsd1 folder

    result_10k, metric, query_list, tboxes_list = histogram_sequence(row, column)
    print('metric_iou_mean = ', metric*100, '%')
    
    
    #write the results in a .pkl file
    pickle_file = '{}/query{}/method{}/result.pkl'.format(results_dir, query_set, method)
    f = open(pickle_file, 'wb')
    pickle.dump((qsd1, result_10k), f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close
    
    pickle_file = '{}/query{}/method{}/text_boxes.pkl'.format(results_dir, query_set, method)
    f = open(pickle_file, 'wb')
    pickle.dump((tboxes_list, query_list), f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close

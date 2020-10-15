import pickle
import time

import cv2
import ml_metrics as metrics

from query_images import BackgroundRemove, HistogramDistance, HistogramGenerator


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
        image = 'qsd2_w1/{:05d}.jpg'.format(i)

        count = 1
        qs = cv2.imread(image)
        qs, mask = BackgroundRemove.remove_background(qs, 2)
        maskbis.append(mask)
        qs2.append(HistogramGenerator.create_hists(qs))
        count = 0

    for i in range(range_bbdd1):
        img = cv2.imread('BBDD/bbdd_{:05d}.jpg'.format(i))
        bbdd.append(HistogramGenerator.create_hists(img))

    for i in range(range_mask2):
        image = 'qsd2_w1/{:05d}.png'.format(i)

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
            distance[key] = HistogramDistance.euclidean(h1, bbdd[key])
            # print(distance[key])
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
    t = time.time() - t
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

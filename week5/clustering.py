import pickle
import numpy as np
import math

import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from image_processing import ImageDescriptors

dataset = open('../BBDD/relationships.pkl', 'rb')
dataset = pickle.load(dataset)

descriptor = []
for i in range(len(dataset)):
    img = cv2.imread('../BBDD/bbdd_{:05d}.jpg'.format(i))
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #histo = cv2.calcHist(HSV, [2], None, [32], (0, 256))
    #descriptor.append([np.mean(HSV[:,:,2])])
    # descriptor.append(ImageDescriptors.histo_wavelet_transform(img, (10, 10)))
    # descriptor.append(ImageDescriptors.hog(img))
    # descriptor.append(cv2.calcHist())
    descriptor.append(ImageDescriptors.histo_wavelet_transform(img, (10, 10)))

# KMeans
n = 10
bests = 5
cols = 2
rows = math.ceil(bests / cols)
kmeans = KMeans(n, max_iter=500)
results = kmeans.fit_transform(descriptor)
predictions = kmeans.predict(descriptor)

for i in range(n):
    items_cluster = np.where(predictions == i)
    results_cluster = results.copy()
    results_cluster[np.where(predictions != i)] = -1
    argsResult = results_cluster[:, i].argsort()
    bestArgs = argsResult[-5:]
    bestArgs = bestArgs[np.isin(bestArgs, items_cluster)]

    fig = plt.figure()
    plt.title('Cluster {}'.format(i))
    plt.axis('off')
    for j in range(len(bestArgs)):
        a = fig.add_subplot(rows, cols, j + 1)
        a.axis('off')
        img = cv2.imread('../BBDD/bbdd_{:05d}.jpg'.format(bestArgs[j]))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.show()

"""

Usage:
  cbir.py <weekNumber> <teamNumber> <querySet> <MethodNumber> <distanceMeasure> [--testDir=<td>]
  cbir.py -h | --help

  <weekNumber> --> Number of the week
  <teamNumber> --> Team Number, in our case 04
  <querySet> --> number of the query
  <MethodNumber> --> Number of the method : 1: Divided Histogram, 2: 3d Color Histogram
  <distanceMeasure> --> 1: Euclidean distance, 2: x^2 distance

  Example of use --> python cbir.py 1 04 0 1 1 2

"""

import imutils
from docopt import docopt
import pickle
import cv2
import ml_metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

from query_images import Distance
from image_processing import ImageNoise, TextDetection, ImageDescriptors, ImageBackgroundRemoval, Rotation


DB_FOLDER = '../BBDD/'
DATASET_FOLDER = ''
results_dir = "/Users/eudal/PycharmProjects/Master/Team-4/week3"


def get_bbdd(folder):
    bbddfile = open(folder + '/relationships.pkl', 'rb')
    bbdd1 = pickle.load(bbddfile)
    return bbdd1


def get_dataset(folder):
    dataset_file = open('{}/gt_corresps.pkl'.format(folder), 'rb')
    dataset = pickle.load(dataset_file)
    return dataset


def generate_results_multiple_images(dataset, bbdd_descriptors, dataset_descriptors, distance_fn, minMaxValue = 4, reverse=True):
    result_1k = []
    result_5k = []
    result_10k = []
    results_10k2d = []
    min_val = 0

    datasetGen = []
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            datasetGen.append([dataset[i][j]])

    for i in range(len(dataset_descriptors)):
        y2 = []
        # print(dataset_descriptors[i])
        for j in range(len(dataset_descriptors[i])):
            h1 = dataset_descriptors[i][j]
            distance = {}
            for key in range(len(bbdd_descriptors)):
                distanceValue = distance_fn(h1, bbdd_descriptors[key])
                #print(distanceValue)
                if reverse:
                    if len(distanceValue) < minMaxValue:
                        distance[key] = len(distanceValue)
                else:
                    if distanceValue > minMaxValue:
                        distance[key] = distanceValue
            print(max(distance.values()))
            if (len(distance) > 0) & (max(distance.values()) > 900):
                y = sorted(distance, key=distance.get, reverse=reverse)[:10]
                x = sorted(distance, key=distance.get, reverse=reverse)[:5]
                z = sorted(distance, key=distance.get, reverse=reverse)[:1]
                y2.append(y)
                result_10k.append(y)
                result_5k.append(x)
                result_1k.append(z)
            else:
                result_10k.append([-1])
                result_5k.append([-1])
                result_1k.append([-1])
            
        # print(y2)
        results_10k2d.append(y2)

    score_k1 = metrics.mapk(datasetGen, result_1k, 1) * 100
    score_k5 = metrics.mapk(datasetGen, result_5k, 5) * 100
    score_k10 = metrics.mapk(datasetGen, result_10k, 10) * 100
    print(result_10k)
    print(dataset)

    print('Score K1 = ', score_k1, '%')
    print('Score K5 = ', score_k5, '%')
    print('Score K10 = ', score_k10, '%')

    # results_dir = "/Users/eudal/PycharmProjects/Master/Team-4/week3"

    """pickle_file = '{}/query{}/method{}/result.pkl'.format(results_dir, query_set, method)
    f = open(pickle_file, 'wb')
    pickle.dump((results_10k2d), f, protocol=4)
    f.close"""
    
    
#generate results DANI
def generate_results_multiple_images(dataset, bbdd_descriptors, dataset_descriptors, distance_fn):
    result_1k = []
    result_5k = []
    result_10k = []
    results_10k2d = []
    min_val = 0
    
    
    datasetGen = []
    for i in range(50):
        for j in range(len(dataset[i])):
            datasetGen.append([dataset[i][j]])

    for i in range(len(dataset_descriptors)):

        y2 = []
        # print(dataset_descriptors[i])
        for j in range(len(dataset_descriptors[i])):
            h1 = dataset_descriptors[i][j]
            distance = {}
            for key in range(len(bbdd_descriptors)):
                distance[key] = distance_fn(h1, bbdd_descriptors[key])
                if distance[key] < 18:
                    del distance[key]
                    distance[-1] = 0
                    
                    
            print(distance)   
            y = sorted(distance, key=distance.get, reverse=True)[:10]
            x = sorted(distance, key=distance.get, reverse=True)[:5]
            z = sorted(distance, key=distance.get, reverse=True)[:1]
            y2.append(y)
            result_10k.append(y)
            result_5k.append(x)
            result_1k.append(z)
        # print(y2)
        results_10k2d.append(y2)

    score_k1 = metrics.mapk(datasetGen, result_1k, 1) * 100
    score_k5 = metrics.mapk(datasetGen, result_5k, 5) * 100
    score_k10 = metrics.mapk(datasetGen, result_10k, 10) * 100
    print(result_10k)
    print(dataset)
    

    print('Score K1 = ', score_k1, '%')
    print('Score K5 = ', score_k5, '%')
    print('Score K10 = ', score_k10, '%')

    # results_dir = "/Users/eudal/PycharmProjects/Master/Team-4/week3"

    pickle_file = '{}/query{}/method{}/result.pkl'.format(results_dir, query_set, method)
    f = open(pickle_file, 'wb')
    pickle.dump((results_10k2d), f, protocol=4)
    f.close

def histogram_noise_qsd2(dataset, descriptor):
dataset_descriptors = []
    c = 0
    for i in range(50):
        img = cv2.imread(DATASET_FOLDER + '/{:05d}.jpg'.format(i))

        img = ImageNoise.remove_noise(img, ImageNoise.MEDIAN)
        img ,rad = Rotation.correct_orientation(img)

        images = ImageBackgroundRemoval.canny(img)
        #print(str(i)+" "+str(len(images)))
        c += len(images)
        # Delete previous iterations authors
        if os.path.exists("./text"+'/{:05d}.txt'.format(i)):
            os.remove("./text"+'/{:05d}.txt'.format(i))
        # Generate descriptors
        descriptorsxImage = []
        for image in images:
            #coordinates, mask = TextDetection.text_detection(image)
            #image[int(coordinates[1] - 5):int(coordinates[3] + 5), int(coordinates[0] - 5):int(coordinates[2] + 5)] = 0
            image = TextDetection.text_detection2(image, '{:05d}'.format(i))
            #image[mask != 0] = 0
            descriptorsxImage.append(ImageDescriptors.generate_descriptor(image, descriptor))
            rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            #plt.imshow(rgb)
            #plt.show()

        dataset_descriptors.append(descriptorsxImage)

    print("TOTALL "+str(c))

    # Generate results
    return dataset_descriptors


def generate_descriptors(dataset, method=1, descriptor=1):
    # Choose method to preprocess and pass descriptor id
    if method == 1:
        dataset_descriptors = histogram_noise_qsd2(dataset, descriptor)

    return dataset_descriptors


def generate_db_descriptors(bbdd, descriptor=1):
    # DB Descriptors
    bbdd_descriptors = []
    bbdd_descriptors1 = []
    bbdd_descriptors2 = []
    for i in range(len(bbdd)):
        if descriptor == ImageDescriptors.TEXT:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            # text = text.readline().split(',')[0].replace('(', '').replace('\'', '')
            author_title = text.readline().split(',')
            if len(author_title) == 0:
                author = ''
                title = ''
            elif len(author_title) < 2:
                author = author_title[0]
                title = ''
            else:
                author = author_title[0].replace('(', '').replace('\'', '')
                title = author_title[1].replace(')', '').replace('\'', '')
            bbdd_descriptors.append(author.replace('\n', ''))
        # TEXT + COLOR
        elif method == 6:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            text = text.readline().split(',')[0].replace('(', '').replace('\'', '');
            bbdd_descriptors2.append(text)

            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors1.append(ImageDescriptors.generate_descriptor(img, ImageDescriptors.HISTOGRAM_CELL))
        # TEXT + TEXTURE_WAVELET
        elif method == 7:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            text = text.readline().split(',')[0].replace('(', '').replace('\'', '');
            bbdd_descriptors2.append(text)

            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors1.append(ImageDescriptors.generate_descriptor(img, ImageDescriptors.TEXTURE_WAVELET))
        # TEXT+COLOR+TEXTURE
        elif method == 8:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            text = text.readline().split(',')[0].replace('(', '').replace('\'', '');
            bbdd_descriptors2.append(text)

            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors1.append(
                ImageDescriptors.generate_descriptor(img, ImageDescriptors.HISTOGRAM_TEXTURE_WAVELET))
        # TEXT+COLOR+ QSD2
        elif method == 9:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            text = text.readline().split(',')[0].replace('(', '').replace('\'', '');
            bbdd_descriptors2.append(text)

            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors1.append(ImageDescriptors.generate_descriptor(img, ImageDescriptors.HISTOGRAM_CELL))
        # TEXT+COLOR+TEXTURE QSD2
        elif method == 10:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            text = text.readline().split(',')[0].replace('(', '').replace('\'', '');
            bbdd_descriptors2.append(text)

            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors1.append(
                ImageDescriptors.generate_descriptor(img, ImageDescriptors.HISTOGRAM_TEXTURE_WAVELET))
        else:
            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors.append(ImageDescriptors.generate_descriptor(img, descriptor))
            # print('text = ', bbdd_descriptors)
    if method > 5:
        return bbdd_descriptors1, bbdd_descriptors2
    else:
        return bbdd_descriptors


if __name__ == "__main__":
    args = docopt(__doc__)

    week = int(args['<weekNumber>'])  # 1
    team = int(args['<teamNumber>'])  # 04
    query_set = int(args['<querySet>'])  # 1 or 2
    method = int(args['<MethodNumber>'])  # 1: divided_hist  2:rgb_3d
    distance_m = int(args['<distanceMeasure>'])  # 1: euclidean and 2: x^2 distance

    DATASET_FOLDER = 'qsd{}_w{}'.format(query_set, week);
    dataset = get_dataset(DATASET_FOLDER)
    bbdd = get_bbdd(DB_FOLDER)

    # Config
    '''
    HISTOGRAM_CELL --> method = 1
    HISTOGRAM_TEXTURE_WAVELET --> method = 1
    TEXT --> method = 2
    TEXTURE_LOCAL_BINARY --> method = 3 (Te problemes )
    TEXTURE_WAVELET --> method = 3 (Te problemes)
    HISTOGRAM_TEXTURE_WAVELET QSD2 --> method = 4
    HISTOGRAM_CELL QSD2 --> method = 4
    TEXTURE_WAVELET QSD2 --> method = 5
    HISTOGRAM_TEXT --> method = 6
    TEXTURE_WAVELET_TEXT --> method = 7
    HISTOGRAM_TEXT_TEXTURE --> method = 8
    HISTOGRAM_TEXT QSD2 --> method = 9
    HISTOGRAM_TEXT_TEXTURE QSD2 --> method = 10
    '''

    descriptor = ImageDescriptors.ORB
    distanceFn = Distance.BFMatcher
    method = 1

    # Call to the test
    print('Generating dataset descriptors')
    dataset_descriptors = generate_descriptors(dataset, method, descriptor)

    # print(dataset_descriptors)
    # print(dataset_descriptors1)
    print('Generating ddbb descriptors')
    bbdd_descriptors = generate_db_descriptors(bbdd, descriptor)

    # print(bbdd_descriptors)
    # print(bbdd_descriptors1)
    # Generate results
    print('Generating results')
    generate_results_multiple_images(dataset, bbdd_descriptors, dataset_descriptors, distanceFn, minMaxValue=9999999)

    # pickle_file = '{}/query{}/method{}/result.pkl'.format(results_dir, query_set, method)
    # f = open(pickle_file, 'wb')
    # pickle.dump((result_10k_pkl), f, protocol=4)
    # f.close

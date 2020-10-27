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

from docopt import docopt
import pickle
import cv2
import ml_metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

from query_images import DescriptorsGenerator, Distance
from image_processing import ImageNoise, TextDetection

DB_FOLDER = '../BBDD'
DATASET_FOLDER = ''

def get_bbdd(folder):
    bbddfile = open(folder + '/relationships.pkl', 'rb')
    bbdd1 = pickle.load(bbddfile)
    return bbdd1


def get_dataset(folder):
    dataset_file = open('{}/gt_corresps.pkl'.format(folder), 'rb')
    dataset = pickle.load(dataset_file)
    return dataset


def background_removal_test():
    pass


def generate_results(dataset, bbdd_descriptors, dataset_descriptors, distance_fn):

    result_1k = []
    result_5k = []
    result_10k = []
    min_val = 0

    for i in range(len(dataset_descriptors)):

        h1 = dataset_descriptors[i]
        distance = {}
        for key in range(len(bbdd_descriptors)):
            distance[key] = distance_fn(h1, bbdd_descriptors[key])
            min_val = min(distance.values())

        x = sorted(distance, key=distance.get, reverse=False)[:5]
        result_5k.append(x)
        y = sorted(distance, key=distance.get, reverse=False)[:10]
        result_10k.append(y)
        result = [key for key, value in distance.items() if value == min_val]
        result_1k.append(result)

    score_k1 = metrics.mapk(dataset, result_1k, 1) * 100
    score_k5 = metrics.mapk(dataset, result_5k, 5) * 100
    score_k10 = metrics.mapk(dataset, result_10k, 10) * 100

    print('Score K1 = ', score_k1, '%')
    print('Score K5 = ', score_k5, '%')
    print('Score K10 = ', score_k10, '%')


def evaluate_noise():
    pass


def histogram_noise(dataset, descriptor):

    dataset_descriptors = []
    for i in range(len(dataset)):
        img = cv2.imread(DATASET_FOLDER+'/{:05d}.jpg'.format(i))
        imgWithoutNoise = cv2.imread(DATASET_FOLDER + '/non_augmented/{:05d}.jpg'.format(i))

        # Preprocess pipeline
        img = ImageNoise.remove_noise(img, ImageNoise.MEDIAN)
        print(cv2.PSNR(imgWithoutNoise, img))
        print(Distance.euclidean(imgWithoutNoise, img))
        evaluate_noise()#//TODO: Implement evaluation of noise

        coordinates, mask = TextDetection.text_detection(img)
        img[int(coordinates[1] - 5):int(coordinates[3] + 5), int(coordinates[0] - 5):int(coordinates[2] + 5)] = 0

        # Generate descriptors
        dataset_descriptors.append(DescriptorsGenerator.generate_descriptor(img, descriptor))

    # Generate results
    return dataset_descriptors

def text_noise(dataset, descriptor):

    dataset_descriptors = []
    for i in range(len(dataset)):
        img = cv2.imread(DATASET_FOLDER+'/{:05d}.jpg'.format(i))

        # Preprocess pipeline
        img = ImageNoise.remove_noise(img, ImageNoise.MEDIAN)
        coordinates, mask = TextDetection.text_detection(img)
        cropped = img[int(coordinates[1] - 5):int(coordinates[3] + 5), int(coordinates[0] - 5):int(coordinates[2] + 5)]

        # Generate descriptors
        dataset_descriptors.append(DescriptorsGenerator.generate_descriptor(cropped, descriptor))

    # Generate results
    return dataset_descriptors

def generate_descriptors(dataset, method=1, descriptor=1):

    #Choose method to preprocess and pass descriptor id
    if method == 1:
        dataset_descriptors = histogram_noise(dataset, descriptor)
    elif method == 2:
        dataset_descriptors = text_noise(dataset, descriptor)

    return dataset_descriptors

def generate_db_descriptors(bbdd, descriptor=1):

    # DB Descriptors
    bbdd_descriptors = []
    for i in range(len(bbdd)):
        if descriptor == DescriptorsGenerator.HISTOGRAM_CELL:
            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors.append(DescriptorsGenerator.generate_descriptor(img, descriptor))

        elif descriptor == DescriptorsGenerator.TEXT:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            text = text.readline();
            bbdd_descriptors.append(text)

    return bbdd_descriptors;


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
    descriptor = DescriptorsGenerator.TEXT
    distanceFn = Distance.levenshtein
    method = 2

    # Call to the test
    print('Generating dataset descriptors')
    dataset_descriptors = generate_descriptors(dataset, method, descriptor)

    print('Generating ddbb descriptors')
    bbdd_descriptors = generate_db_descriptors(bbdd, descriptor)

    # Generate results
    print('Generating results')
    generate_results(dataset, bbdd_descriptors, dataset_descriptors, distanceFn)

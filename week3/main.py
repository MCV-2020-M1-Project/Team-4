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

from query_images import Distance
from image_processing import ImageNoise, TextDetection, ImageDescriptors, ImageBackgroundRemoval

DB_FOLDER = '../BBDD/'
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
    print(result_10k)
    print(dataset)

    print('Score K1 = ', score_k1, '%')
    print('Score K5 = ', score_k5, '%')
    print('Score K10 = ', score_k10, '%')

def generate_results_multiple_images(dataset, bbdd_descriptors, dataset_descriptors, distance_fn):

    result_1k = []
    result_5k = []
    result_10k = []
    min_val = 0

    datasetGen = []

    for i in range(len(dataset_descriptors)):
        for j in range(len(dataset_descriptors[i])):
            h1 = dataset_descriptors[i][j]
            datasetGen.append([dataset[i][j]])
            distance = {}
            for key in range(len(bbdd_descriptors)):
                distance[key] = distance_fn(h1, bbdd_descriptors[key])

            y = sorted(distance, key=distance.get, reverse=False)[:10]
            x = sorted(distance, key=distance.get, reverse=False)[:5]
            z = sorted(distance, key=distance.get, reverse=False)[:1]

            result_10k.append(y)
            result_5k.append(x)
            result_1k.append(z)

    score_k1 = metrics.mapk(datasetGen, result_1k, 1) * 100
    score_k5 = metrics.mapk(datasetGen, result_5k, 5) * 100
    score_k10 = metrics.mapk(datasetGen, result_10k, 10) * 100
    print(result_10k)
    print(dataset)

    print('Score K1 = ', score_k1, '%')
    print('Score K5 = ', score_k5, '%')
    print('Score K10 = ', score_k10, '%')

def evaluate_noise():
    pass


def simple_descriptor(dataset, descriptor):

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
        dataset_descriptors.append(ImageDescriptors.generate_descriptor(img, descriptor))

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
        try:
            cv2.imwrite('textos{}.png'.format(i),cropped)
        except:
            pass
        # Generate descriptors
        dataset_descriptors.append(ImageDescriptors.generate_descriptor(cropped, descriptor))

    # Generate results
    return dataset_descriptors

def texture_descriptors(dataset, descriptor):
    dataset_descriptors = []
    for i in range(len(dataset)):
        img = cv2.imread(DATASET_FOLDER+'/{:05d}.jpg'.format(i),0)
        # Generate descriptors    
        dataset_descriptors.append(ImageDescriptors.generate_descriptor(img, descriptor))
        # Generate results
    return dataset_descriptors


def histogram_noise_qsd2(dataset, descriptor):

    dataset_descriptors = []
    for i in range(len(dataset)):
        img = cv2.imread(DATASET_FOLDER + '/{:05d}.jpg'.format(i))

        img = ImageNoise.remove_noise(img, ImageNoise.AVERAGE)
        images = ImageBackgroundRemoval.canny(img)

        # Generate descriptors
        descriptorsxImage = []
        for image in images:
            coordinates, mask = TextDetection.text_detection(image)
            image[int(coordinates[1] - 5):int(coordinates[3] + 5), int(coordinates[0] - 5):int(coordinates[2] + 5)] = 0
            descriptorsxImage.append(ImageDescriptors.generate_descriptor(image, descriptor))

        dataset_descriptors.append(descriptorsxImage)

    # Generate results
    return dataset_descriptors

def texture_noise_qsd2(dataset, descriptor):

    dataset_descriptors = []
    for i in range(len(dataset)):
        img = cv2.imread(DATASET_FOLDER + '/{:05d}.jpg'.format(i))

        img = ImageNoise.remove_noise(img, ImageNoise.AVERAGE)
        images = ImageBackgroundRemoval.canny(img)

        # Generate descriptors
        descriptorsxImage = []
        for image in images:
            coordinates, mask = TextDetection.text_detection(image)
            image[int(coordinates[1] - 5):int(coordinates[3] + 5), int(coordinates[0] - 5):int(coordinates[2] + 5)] = 0
            descriptorsxImage.append(ImageDescriptors.generate_descriptor(image, descriptor))

        dataset_descriptors.append(descriptorsxImage)

    # Generate results
    return dataset_descriptors


def generate_descriptors(dataset, method=1, descriptor=1):

    #Choose method to preprocess and pass descriptor id
    if method == 1:
        dataset_descriptors = simple_descriptor(dataset, descriptor)
    elif method == 2:
        dataset_descriptors = text_noise(dataset, descriptor)
    elif method == 3:
        dataset_descriptors = texture_descriptors(dataset, descriptor)
    elif method == 4:
        dataset_descriptors = histogram_noise_qsd2(dataset, descriptor)
    elif method == 5:
        dataset_descriptors = texture_noise_qsd2(dataset, descriptor)

    return dataset_descriptors

def generate_db_descriptors(bbdd, descriptor=1):

    # DB Descriptors
    bbdd_descriptors = []
    for i in range(len(bbdd)):
        if descriptor == ImageDescriptors.TEXT:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            #text = text.readline().split(',')[0].replace('(', '').replace('\'', '')
            author_title = text.readline().split(',')
            if len(author_title) == 0:
                author = ''
                title = ''
            elif len(author_title) < 2:
                author = author_title[0]
                title = ''
            else:
                author = author_title[0].replace('(','').replace('\'','')
                title = author_title[1].replace(')','').replace('\'','')
            bbdd_descriptors.append(author.replace('\n',''))
        else:
            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors.append(ImageDescriptors.generate_descriptor(img, descriptor))
    print('text = ', bbdd_descriptors)
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
    descriptor = ImageDescriptors.TEXT
    distanceFn = Distance.levenshtein
    method = 2

    # Call to the test
    print('Generating dataset descriptors')
    dataset_descriptors = generate_descriptors(dataset, method, descriptor)
    print(dataset_descriptors)

    print('Generating ddbb descriptors')
    bbdd_descriptors = generate_db_descriptors(bbdd, descriptor)
    print(bbdd_descriptors)
    # Generate results
    print('Generating results')
    if query_set == 1:
        generate_results(dataset, bbdd_descriptors, dataset_descriptors, distanceFn)
    elif query_set == 2:
        generate_results_multiple_images(dataset, bbdd_descriptors, dataset_descriptors, distanceFn)

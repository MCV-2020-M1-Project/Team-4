"""

Usage:
  cbir.py <weekNumber> <teamNumber> <querySet> <MethodNumber> <distanceMeasure> [--testDir=<td>]
  cbir.py -h | --help

  <weekNumber> --> Number of the week
  <teamNumber> --> Team Number, in our case 04
  <querySet> --> number of the query
  <MethodNumber> --> Number of the method : 1: Divided Histogram, 2: 3d Color Histogram
  <distanceMeasure> --> 1: Euclidean distance, 2: x^2 distance

  Example of use --> python main.py 1 04 1 1 2

"""
import imutils
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
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            datasetGen.append([dataset[i][j]])

    for i in range(len(dataset_descriptors)):
        for j in range(len(dataset_descriptors[i])):
            h1 = dataset_descriptors[i][j]
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


def generate_results_two_descriptors(dataset, bbdd_descriptors, dataset_descriptors, distance_fn, bbdd_descriptors2, dataset_descriptors2, distance_fn2):
    result_1k = []
    result_5k = []
    result_10k = []
    min_val = 0
    
    for i in range(len(dataset_descriptors)):
        distance = {}
        for key in range(len(bbdd_descriptors)):
            h1 = dataset_descriptors[i]
            distance[key] = distance_fn(h1, bbdd_descriptors[key])
            min_val = min(distance.values())
        
        for key in range(len(bbdd_descriptors2)):
            h1 = dataset_descriptors2[i]
            distance[key] = distance[key] + distance_fn2(h1, bbdd_descriptors2[key])
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
    
def generate_results_multiple_images_two_descriptors(dataset, bbdd_descriptors, dataset_descriptors, distance_fn, bbdd_descriptors2, dataset_descriptors2, distance_fn2):

    result_1k = []
    result_5k = []
    result_10k = []
    min_val = 0
    distance = {}
    datasetGen = []
    n = 0
    for i in range(len(dataset_descriptors)):
        for j in range(len(dataset_descriptors[i])):
            h1 = dataset_descriptors[i][j]
            datasetGen.append([dataset[i][j]])
            for key in range(len(bbdd_descriptors)):
                distance[key] = distance_fn(h1, bbdd_descriptors[key]) 
            h2 = dataset_descriptors2[i]
            for key in range(len(bbdd_descriptors2)):
                distance[key] = distance[key] + distance_fn2(h2, bbdd_descriptors2[key])

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
    print(datasetGen)

    print('Score K1 = ', score_k1, '%')
    print('Score K5 = ', score_k5, '%')
    print('Score K10 = ', score_k10, '%')    
    
def evaluate_noise():
    pass


def simple_descriptor(dataset, descriptor):

    dataset_descriptors = []
    for i in range(len(dataset)):
        img = cv2.imread(DATASET_FOLDER+'/{:05d}.jpg'.format(i))
        ##imgWithoutNoise = cv2.imread(DATASET_FOLDER + '/non_augmented/{:05d}.jpg'.format(i))

        # Preprocess pipeline
        img = ImageNoise.remove_noise(img, ImageNoise.MEDIAN)
        ##print(cv2.PSNR(imgWithoutNoise, img))
        ##print(Distance.euclidean(imgWithoutNoise, img))
        ##evaluate_noise()#//TODO: Implement evaluation of noise

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

def two_descriptors(dataset, descriptor):

    dataset_descriptors1 = []
    dataset_descriptors2 = []
    for i in range(len(dataset)):
        img = cv2.imread(DATASET_FOLDER+'/{:05d}.jpg'.format(i))
        # Preprocess pipeline
        img = ImageNoise.remove_noise(img, ImageNoise.MEDIAN)
        coordinates, mask = TextDetection.text_detection(img)
        cropped = img[int(coordinates[1] - 5):int(coordinates[3] + 5), int(coordinates[0] - 5):int(coordinates[2] + 5)]
        # Generate descriptors
        dataset_descriptors2.append(ImageDescriptors.generate_descriptor(cropped, method = 2))
        
    for i in range(len(dataset)):
        
        img = cv2.imread(DATASET_FOLDER+'/{:05d}.jpg'.format(i))
        # Preprocess pipeline
        img = ImageNoise.remove_noise(img, ImageNoise.MEDIAN)
        # Generate descriptors
        a = ImageDescriptors.generate_descriptor(img, descriptor)
        dataset_descriptors1.append(a)

    # Generate results
    return dataset_descriptors1, dataset_descriptors2 
  
def two_descriptors_qsd2(dataset, descriptor):

    dataset_descriptors1 = []
    dataset_descriptors2 = []
    for i in range(len(dataset)):
        img = cv2.imread(DATASET_FOLDER+'/{:05d}.jpg'.format(i))
        # Preprocess pipeline
        img = ImageNoise.remove_noise(img, ImageNoise.MEDIAN)
        images = ImageBackgroundRemoval.canny(img)
        descriptorsxImage = []
        for image in images:
            coordinates, mask = TextDetection.text_detection(image)
            cropped = image[int(coordinates[1] - 5):int(coordinates[3] + 5), int(coordinates[0] - 5):int(coordinates[2] + 5)]
            descriptorsxImage.append(ImageDescriptors.generate_descriptor(cropped, method = 2))
            dataset_descriptors2.append(descriptorsxImage)
        
    for i in range(len(dataset)):
        
        img = cv2.imread(DATASET_FOLDER+'/{:05d}.jpg'.format(i))
        # Preprocess pipeline
        img = ImageNoise.remove_noise(img, ImageNoise.MEDIAN)
        images = ImageBackgroundRemoval.canny(img)
        # Generate descriptors
        descriptorsxImage2 = []
        for image in images:
            descriptorsxImage2.append(ImageDescriptors.generate_descriptor(image, descriptor))
        
        # Generate descriptors
        dataset_descriptors1.append(descriptorsxImage2)


    # Generate results
    return dataset_descriptors1, dataset_descriptors2
  
def descriptor_noise_qsd2(dataset, descriptor):

    dataset_descriptors = []
    qt = 0
    for i in range(len(dataset)):
        img = cv2.imread(DATASET_FOLDER + '/{:05d}.jpg'.format(i))

        """plt.imshow(img)
        plt.show()"""

        img = ImageNoise.remove_noise(img, ImageNoise.MEDIAN, 5)
        images = ImageBackgroundRemoval.canny(img)
        qt += len(images)

        # Generate descriptors
        descriptorsxImage = []
        for image in images:

            image = imutils.resize(image, width=img.shape[0] // 4)
            coordinates, mask = TextDetection.text_detection(image)
            image[int(coordinates[1] - 5):int(coordinates[3] + 5), int(coordinates[0] - 5):int(coordinates[2] + 5)] = 255
           # plt.imshow(image);plt.show();
            descriptorsxImage.append(ImageDescriptors.generate_descriptor(image, descriptor))

        dataset_descriptors.append(descriptorsxImage)

    print(qt)

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
        dataset_descriptors = descriptor_noise_qsd2(dataset, descriptor)
    elif method > 5:
        if method >8:
            dataset_descriptors1, dataset_descriptors2 = two_descriptors_qsd2(dataset, descriptor)
        else:
            dataset_descriptors1, dataset_descriptors2 = two_descriptors(dataset, descriptor)
    if method > 5:
        return dataset_descriptors1, dataset_descriptors2
    else:
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

        #TEXT + COLOR
        elif method == 6:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            text = text.readline().split(',')[0].replace('(', '').replace('\'', '');
            bbdd_descriptors2.append(text)
            
            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors1.append(ImageDescriptors.generate_descriptor(img, ImageDescriptors.HISTOGRAM_CELL))
        #TEXT + TEXTURE_WAVELET
        elif method == 7:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            text = text.readline().split(',')[0].replace('(', '').replace('\'', '');
            bbdd_descriptors2.append(text)
            
            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors1.append(ImageDescriptors.generate_descriptor(img, ImageDescriptors.TEXTURE_WAVELET))
        
        #TEXT+COLOR+TEXTURE
        elif method == 8:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            text = text.readline().split(',')[0].replace('(', '').replace('\'', '');
            bbdd_descriptors2.append(text)
            
            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors1.append(ImageDescriptors.generate_descriptor(img, ImageDescriptors.HISTOGRAM_TEXTURE_WAVELET))
        #TEXT+COLOR+ QSD2
        elif method == 9:
            text = open(DB_FOLDER + '/bbdd_{:05d}.txt'.format(i), encoding='iso-8859-1')
            text = text.readline().split(',')[0].replace('(', '').replace('\'', '');
            bbdd_descriptors2.append(text)
            
            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors1.append(ImageDescriptors.generate_descriptor(img, ImageDescriptors.HISTOGRAM_CELL))
        else:
            img = cv2.imread(DB_FOLDER + '/bbdd_{:05d}.jpg'.format(i))
            bbdd_descriptors.append(ImageDescriptors.generate_descriptor(img, descriptor))
    print('text = ', bbdd_descriptors)
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
    TEXT --> method = 2
    TEXTURE_COSINE --> method = 3
    TEXTURE_LOCAL_BINARY --> method = 3
    TEXTURE_WAVELET --> method = 3
    HISTOGRAM_TEXTURE_WAVELET --> method = 4
    HISTOGRAM_TEXT --> method = 6
    TEXTURE_WAVELET_TEXT --> method = 7
    HISTOGRAM_TEXT_TEXTURE --> method = 8
    HISTOGRAM_TEXT QSD2 --> method = 9
    
    distanceFn2 = Distance.levenshtein
    '''
    
    descriptor = ImageDescriptors.HISTOGRAM_TEXT
    distanceFn = Distance.x2distance
    method = 4
    distanceFn2 = Distance.levenshtein

    # Call to the test
    print('Generating dataset descriptors')
    if method > 5:
         dataset_descriptors1, bbdd_descriptors2 = generate_descriptors(dataset, descriptor)
    else:
        dataset_descriptors = generate_descriptors(dataset, descriptor)
    #print(dataset_descriptors)

    print('Generating ddbb descriptors')
    if method > 5:
         bbdd_descriptors1, bbdd_descriptors2 = generate_db_descriptors(bbdd, descriptor)
    else:
        bbdd_descriptors = generate_db_descriptors(bbdd, descriptor)
    #print(bbdd_descriptors)
    # Generate results
    print('Generating results')
    if query_set == 1:
        if method > 5:
            generate_results_two_descriptors(dataset, bbdd_descriptors1, dataset_descriptors1, distanceFn, bbdd_descriptors2, dataset_descriptors2, distanceFn2)
        else:
            generate_results(dataset, bbdd_descriptors, dataset_descriptors, distanceFn)
             
    elif query_set == 2:
        if method > 5:
            generate_results_multiple_images_two_descriptors(dataset, bbdd_descriptors1, dataset_descriptors1, distanceFn, bbdd_descriptors2, dataset_descriptors2, distanceFn2)
        else:
            generate_results_multiple_images(dataset, bbdd_descriptors, dataset_descriptors, distanceFn)
        
    

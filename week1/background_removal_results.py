import cv2
import imutils
import math
import pickle
import numpy as np
import ml_metrics as metrics
import time
from docopt import docopt
import statistics as stats


def create_hists(image):
    if count == 0:
        image = cv2.imread(image)
    height, width, dimensions = image.shape
    if height > 250:
        factor = height//250
        image = cv2.resize(image, (width//factor, height//factor), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    height, width = img.shape
    # Cut the image in 4
    column = 4
    row = 16
    height_cutoff = height // row
    width_cutoff = width // column
    outputArray = []

    dimensions = 3
    for d in range(dimensions):
        for c in range(column):
            for r in range(row):
                s1 = image[r * height_cutoff:(r + 1) * height_cutoff, c * width_cutoff: (c + 1) * width_cutoff, d]
                s1_hist = np.array(cv2.calcHist([s1], [0], None, [32], [0, 256]))
                cv2.normalize(s1_hist, s1_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                outputArray = np.concatenate((outputArray, s1_hist), axis=None)
    return outputArray

def rgb_hist_3d(image, bins=8, mask=None):
    
    if count == 0:
        image = cv2.imread(image)
    hist = cv2.calcHist([image], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()


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



# Class to remove backgrounds and evaluate masks
class BackgroundRemove(object):
    __count = 1

    # Method 1 using Edge detector
    @staticmethod
    def method1(image, show_output=False):

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize
        resize = imutils.resize(gray, width=300)
        ratio = image.shape[0] / resize.shape[0]

        # Apply gaussian and edges
        gaussiana = cv2.GaussianBlur(resize, (3, 3), 0)
        edges = cv2.Canny(gaussiana, 50, 200)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

        # Contours detector
        (contours, hierarchy) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Generate mask
        mask = BackgroundRemove.generate_mask(image, contours, ratio)

        if show_output:
            cv2.imshow('GAU', gaussiana)
            cv2.imshow('Edges', edges)
            cv2.imshow('Image', gray)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return mask

    # Method 2 using Threshold and Morphological detector
    @staticmethod
    def method2(image, morph_type=2, show_output=False):

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize
        resize = imutils.resize(gray, width=500)
        ratio = image.shape[0] / resize.shape[0]

        # Apply Gaussian, Sharpness, Threshold and Morphological transformations
        gaussian = cv2.GaussianBlur(resize, (3, 3), 0)
        sharpness = cv2.filter2D(gaussian, cv2.CV_8U, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        threshold = cv2.threshold(sharpness, 90, 255, cv2.THRESH_BINARY_INV)[1]
        morphological = BackgroundRemove.apply_morph(threshold, morph_type)

        # Detect contours
        (contours, hierarchy) = cv2.findContours(morphological, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Generate mask
        mask = BackgroundRemove.generate_mask(image, contours, ratio)

        if show_output:
            cv2.imshow('Image', resize)
            cv2.imshow('Image Tresh', morphological)
            cv2.imshow('Mask', mask)
            cv2.imwrite(str(BackgroundRemove.__count) + ".jpg", gray)

            cv2.imwrite("gray.jpg", gray)
            cv2.imwrite("gaussian.jpg", gaussian)
            cv2.imwrite("sharpness.jpg", sharpness)
            cv2.imwrite("treshold.jpg", threshold)
            cv2.imwrite("morph.jpg", morphological)
            cv2.imwrite("result.jpg", BackgroundRemove.crop_with_mask(image, mask))

            BackgroundRemove.__count = BackgroundRemove.__count + 1

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return mask

    # Method3 Color treshold
    @staticmethod
    def method3(image, show_output=False):

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)[1]
        (contours, hierarchy) = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = BackgroundRemove.generate_mask(image, contours, 1)

        if show_output:
            cv2.imshow('GAU', gray)
            cv2.imshow('Treshold', threshold)
            cv2.imshow('Mask', mask)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return mask

    # Apply to a image different morphological transformations and returns this one
    @staticmethod
    def apply_morph(image, morph_type):

        # Method1: Two closes of different size
        if morph_type == 1:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                     iterations=1)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
                                     iterations=3)

        # Method2: One close in x axis
        if morph_type == 2:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=np.uint8), iterations=3)

        # Method3: One close with CROSS structure and 15 iterations
        if morph_type == 3:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5), dtype=np.uint8),
                                     iterations=15)

        return image

    @staticmethod
    def generate_mask(image, contours, ratio):

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Detect the biggest contour and draw this in the mask image
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(mask, [c], -1, (255, 0, 0), -1)

        return mask

    @staticmethod
    def crop_with_mask(image, mask):
        rect = cv2.boundingRect(mask)
        cropped_img = image[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        return cropped_img

    # Main method to remove a background, returns the image cropped and the mask
    @staticmethod
    def remove_background(image, method=1, show_output=False):

        # Apply method to generate mask
        if method == 1:# Edges
            mask = BackgroundRemove.method1(image, show_output)
        elif method == 2:# Morphological
            mask = BackgroundRemove.method2(image, 2, show_output) #
        elif method == 3:# Threshold
            mask = BackgroundRemove.method3(image, show_output)

        # Generate image cropped
        image_crop = BackgroundRemove.crop_with_mask(image, mask)

        if show_output:
            cv2.imshow('Image Result', image_crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

        return image_crop, mask

    # Method to evaluate the mask, returns precision, recall and f1-score indexes
    @staticmethod
    def evaluate_mask(original_mask, generated_mask):

        # Calculate True-Positive, False-Positive and False-Negative
        tp = np.sum(np.logical_and(original_mask, generated_mask))
        fp = np.sum(np.logical_and(np.logical_not(original_mask), generated_mask))
        fn = np.sum(np.logical_and(original_mask, np.logical_not(generated_mask)))

        # Calculate precision, recall and f1-score
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        return precision, recall, f1_score
    
    
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
    maskbis = [] # Array (np.array) with the histogram of each image in the mask folder
    count=0
    
    
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
    

        qs1.append(create_hists(image))
        
    
    for i in range(range_qsd2):
        if i < 10:
            image = 'qsd2_w1/0000' + str(i) + '.jpg'
        else:
            image = 'qsd2_w1/000' + str(i) + '.jpg'
    
        count=1
        qs = cv2.imread(image)
        qs, mask = BackgroundRemove.remove_background(qs,2)
        maskbis.append(mask)
        qs2.append(create_hists(qs))
        count=0

    for i in range(range_bbdd1):
        if i < 10:
            image = 'BBDD/bbdd_0000' + str(i) + '.jpg'
        elif i < 100:
            image = 'BBDD/bbdd_000' + str(i) + '.jpg'
        else:
            image = 'BBDD/bbdd_00' + str(i) + '.jpg'
            

        bbdd.append(create_hists(image))

    
    for i in range(range_mask2):
        if i < 10:
            image = 'qsd2_w1/0000' + str(i) + '.png'


        elif i < 100:
            image = 'qsd2_w1/000' + str(i) + '.png'
           
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
            distance[key] = euclidean(h1, bbdd[key])
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

    score_k1 = metrics.mapk(qsd2, result_1k, 1) * 100
    score_k5 = metrics.mapk(qsd2, result_5k, 5) * 100
    score_k10 = metrics.mapk(qsd2, result_10k, 10) * 100

    print('Score K1 = ', score_k1, '%')
    print('Score K5 = ', score_k5, '%')
    print('Score K10 = ', score_k10, '%')
    t = time.time()-t
    print("time needed to complete sequence: ", t)
    print("for each image (aprox): ", t / range_qsd1)
    
 '''   
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

'''    

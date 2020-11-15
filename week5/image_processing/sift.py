import cv2 as cv2
import matplotlib.pyplot as plt
import pickle
import math
from image_processing import ImageNoise, ImageNoise


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects


bbddfile = open('BBDD/relationships.pkl', 'rb')
bbdd1 = pickle.load(bbddfile)

dataset_file = open('qsd1_w3/gt_corresps.pkl', 'rb')
dataset = pickle.load(dataset_file)

#sift
sift = cv2.xfeatures2d.SIFT_create(nfeatures=2000)

list_match = {}
result_10k = []

descriptors_2_tot = init_list_of_objects(len(bbdd1))
for i in range(len(bbdd1)):
    keypoints_2 = [[]]
    img2 = cv2.imread('BBDD/bbdd_{:05d}.jpg'.format(i),0)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
    descriptors_2_tot[i] = descriptors_2

for j in range(len(dataset)):
    # read images
    
    img1 = cv2.imread('qsd1_w3/{:05d}.jpg'.format(j),0)
    img1 = ImageNoise.remove_noise(img1, ImageNoise.MEDIAN)
    scale_percent = 50 # percent of original size
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    if img1.shape[0] > 500 or img1.shape[1]  > 500:
        img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
        
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    for i in range(len(bbdd1)):
        print('here', i)
        count = 0
        try:

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors_1, descriptors_2_tot[i], k=2)
            good=[]
            for m, n in matches:
                if m.distance < 0.5 * n.distance:
                    good.append([m])
                    count += 1
        except:
            pass
        list_match[i] = count
    x = sorted(list_match, key=list_match.get, reverse=True)[:10]
    result_10k.append(x)
    print(j, x)

print(result_10k)

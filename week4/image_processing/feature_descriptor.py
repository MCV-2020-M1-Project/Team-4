import cv2 
import numpy as np 
from image_processing import ImageNoise, ImageDescriptors
from skimage.feature import local_binary_pattern as lbp

class Feature_Descriptor(object):
    
    def lbp_feature_descriptor(image, vector):
        image = ImageNoise.remove_noise(image, ImageNoise.MEDIAN)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        hist_keypoint = []
        for i in range(len(vector)):
            n = vector[i][0]
            m = vector[i][1]
            try:
                matrix = image[n-5:n+5, m-5:m+5]
                radius = 2
                no_points = 8 * radius
                img2 = lbp(matrix, no_points, radius)
                histogram, bins = np.histogram(img2.reshape(-1), 32, range=[0, 256])
                normalize = np.linalg.norm(histogram)
                full_histogram = histogram/normalize
            
                hist_keypoint.append(full_histogram)
            except:
                pass
            
        return hist_keypoint

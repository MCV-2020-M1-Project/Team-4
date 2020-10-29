import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import os
import skimage
from skimage.feature import local_binary_pattern as lbp
from scipy.fftpack import fft, dct
import pywt
from image_processing.image_utils import ImageUtils


class ImageDescriptors(object):
    
    #Resultados muy bajos, falta lo del zigzag que no lo acabo de ver.
    @staticmethod
    def cosine_transform(img):
        img = ImageUtils.divide_image(img, 10, 10)
        img2 = img[:]
        full_histogram = []
        for i in range(len(img)):
            for j in range(len(img[i])):
                #DCT per block
                img2 = (dct(img[i][j],1))
                #ZigZag?¿
                #vector = np.concatenate([np.diagonal(img2[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-img2.shape[0], img2.shape[0])])
                #vector = vector[0:100]
                histogram, bins = np.histogram(img2.reshape(-1), 32, range=[0, 256])
                normalize = np.linalg.norm(histogram)
                full_histogram = np.concatenate((full_histogram, (histogram/normalize)))

        return full_histogram
    
    @staticmethod
    def local_binary_p(img):
        img = ImageUtils.divide_image(img, 10, 10)
        img2 = img[:]
        full_histogram = []
        radius = 2
        no_points = 8 * radius
        for i in range(len(img)):
            for j in range(len(img[i])):
                #lbp per block
                img2 = (lbp(img[i][j], no_points, radius))
                histogram, bins = np.histogram(img2.reshape(-1), 32, range=[0, 256])
                normalize = np.linalg.norm(histogram)
                full_histogram = np.concatenate((full_histogram, (histogram/normalize)))
                
        return full_histogram
    
    #No funciona...
    @staticmethod
    def wavelet_transform(img):
        # Wavelet transform of image
        img = ImageUtils.divide_image(img, 4, 16)
        full_histogram = []
        img2 = img[:]
        for i in range(len(img)):
            for j in range(len(img[i])):
                coeffs2 = pywt.dwt2(img[i][j], 'haar')
                img2, (LH, HL, HH) = coeffs2
                histogram, bins = np.histogram(img2.reshape(-1), 32, range=[0, 256])
                normalize = np.linalg.norm(histogram)
                full_histogram = np.concatenate((full_histogram, (histogram/normalize)))

        return full_histogram

import imutils
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import os
import skimage
import pytesseract
from skimage import feature
from skimage.feature import local_binary_pattern as lbp
from scipy.fftpack import fft, dct
import pywt
from image_processing.image_utils import ImageUtils
from skimage.feature import hog

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import os
import skimage
import pytesseract
from skimage.feature import local_binary_pattern as lbp
from scipy.fftpack import fft, dct
import pywt
from image_processing.image_utils import ImageUtils
from image_processing.image_noise import ImageNoise
from image_processing.text_detection import TextDetection


class ImageDescriptors(object):
    HISTOGRAM_CELL = 1
    TEXT = 2
    TEXTURE_COSINE = 3
    TEXTURE_LOCAL_BINARY = 4
    TEXTURE_WAVELET = 5
    HISTOGRAM_TEXTURE_WAVELET = 6
    HISTOGRAM_TEXT = 7
    TEXTURE_WAVELET_TEXT = 8
    HISTOGRAM_TEXT_TEXTURE = 9

    # Local descriptors descriptors
    SIFT = 10
    HOG = 11
    LBP = 12

    @staticmethod
    def generate_descriptor(image, method=1, cellSize=(10, 10)):
        if method == ImageDescriptors.HISTOGRAM_CELL:
            return ImageDescriptors.histogramCell(image, cellSize)

        elif method == ImageDescriptors.TEXT:
            return ImageDescriptors.getTextFromImage(image)

        elif method == ImageDescriptors.TEXTURE_COSINE:
            return ImageDescriptors.cosine_transform(image)

        elif method == ImageDescriptors.TEXTURE_LOCAL_BINARY:
            return ImageDescriptors.local_binary_p(image)

        elif method == ImageDescriptors.TEXTURE_WAVELET:
            return ImageDescriptors.wavelet_transform(image)

        elif method == ImageDescriptors.HISTOGRAM_TEXTURE_WAVELET:
            return ImageDescriptors.histo_wavelet_transform(image, cellSize)

        elif method == ImageDescriptors.HISTOGRAM_TEXT:
            return ImageDescriptors.histo_text(image, cellSize)

        elif method == ImageDescriptors.TEXTURE_WAVELET_TEXT:
            return ImageDescriptors.texture_text(image, cellSize)

        elif method == ImageDescriptors.HISTOGRAM_TEXT_TEXTURE:
            return ImageDescriptors.texture_text_hist(image, cellSize)

        elif method == ImageDescriptors.SIFT:
            return ImageDescriptors.sift(image)

        elif method == ImageDescriptors.HOG:
            return ImageDescriptors.hog(image)

        elif method == ImageDescriptors.LBP:
            return ImageDescriptors.lbp(image)

    @staticmethod
    def histogramCell(image, cellSize=(10, 10)):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        height, width, dimensions = image.shape

        # Reduce size
        if height > 250:
            factor = height // 250
            image = cv2.resize(image, (width // factor, height // factor), interpolation=cv2.INTER_AREA)

        # Number of divisions
        column, row = cellSize
        output_array = []

        for d in range(dimensions):
            # Adaptable number of bins, to give each dimension more or less weight in the final evaluation
            if d == 0:
                bins = 16
            else:
                bins = 32

            img_divided = ImageUtils.divide_image(image[:, :, d], row, column)
            output_array = np.concatenate((output_array, ImageUtils.calc_hist(img_divided, bins)))

        return output_array

    @staticmethod
    def getTextFromImage(image):
        if image.shape[0] == 0 or image.shape[1] == 0:
            return ""

        return pytesseract.image_to_string(image)

    # Resultados muy bajos, falta lo del zigzag que no lo acabo de ver.
    @staticmethod
    def cosine_transform(img):
        img = ImageUtils.divide_image(img, 10, 10)
        img2 = img[:]
        full_histogram = []
        for i in range(len(img)):
            for j in range(len(img[i])):
                # DCT per block
                img2 = (dct(img[i][j], 1))
                # ZigZag?¿
                # vector = np.concatenate([np.diagonal(img2[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-img2.shape[0], img2.shape[0])])
                # vector = vector[0:100]
                histogram, bins = np.histogram(img2.reshape(-1), 32, range=[0, 256])
                normalize = np.linalg.norm(histogram)
                full_histogram = np.concatenate((full_histogram, (histogram / normalize)))

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
                # lbp per block
                img2 = (lbp(img[i][j], no_points, radius))
                histogram, bins = np.histogram(img2.reshape(-1), 32, range=[0, 256])
                normalize = np.linalg.norm(histogram)
                full_histogram = np.concatenate((full_histogram, (histogram / normalize)))

        return full_histogram

    # Buenos resultados con img2 e img3 (detalles horizontales, detalles verticales)
    @staticmethod
    def wavelet_transform(img):
        # Wavelet transform of image
        img = ImageUtils.divide_image(img, 4, 16)
        full_histogram = []
        img2 = img[:]
        for i in range(len(img)):
            for j in range(len(img[i])):
                coeffs2 = pywt.dwt2(img[i][j], 'haar')
                img1, (img2, img3, img4) = coeffs2
                histogram, bins = np.histogram(img3.reshape(-1), 32, range=[0, 256])
                normalize = np.linalg.norm(histogram)
                full_histogram = np.concatenate((full_histogram, (histogram / normalize)))

        return full_histogram

    @staticmethod
    def histo_wavelet_transform(image, cellSize):
        histoDesc = ImageDescriptors.histogramCell(image, cellSize)
        waveletDesc = ImageDescriptors.wavelet_transform(image)
        return np.concatenate((histoDesc, waveletDesc))

    @staticmethod
    def histo_text(image, cellSize):
        coordinates, mask = TextDetection.text_detection(image)
        image[int(coordinates[1] - 5):int(coordinates[3] + 5), int(coordinates[0] - 5):int(coordinates[2] + 5)] = 0
        histoDesc = ImageDescriptors.histogramCell(image, cellSize)
        return histoDesc

    @staticmethod
    def texture_text(image, cellSize):
        textureDesc = ImageDescriptors.wavelet_transform(image)
        return textureDesc

    @staticmethod
    def texture_text_hist(image, cellSize):
        return ImageDescriptors.histo_wavelet_transform(image, cellSize)

    @staticmethod
    def sift(image):

        sift = cv2.SIFT_create(nfeatures=3000)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        scale_percent = 50  # percent of original size
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        if gray.shape[0] > 500 or gray.shape[1] > 500:
            gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

        keypoints_1, descriptors_1 = sift.detectAndCompute(gray, None)

        return descriptors_1

    @staticmethod
    def hog(image):

        eps = 1e-7
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = imutils.resize(gray, width=gray.shape[1]//4)

        fd = hog(gray, feature_vector=True)

        (hist, _) = np.histogram(fd, bins=8*8, range=(0, 9))

        return hist

    @staticmethod
    def lbp(image):

        eps = 1e-7

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        lbp = feature.local_binary_pattern(gray, 8,  1, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, 8 + 3),
                                 range=(0, 8 + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist

import imutils
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import os
import skimage
import pytesseract
from skimage import feature, data, exposure
from skimage.feature import local_binary_pattern as lbp
from skimage.feature import hog
from scipy.fftpack import fft, dct
import pywt
from image_processing.image_utils import ImageUtils
from skimage.feature import hog


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
            return ImageDescriptors.hog3(image)

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

        sift = cv2.SIFT_create(nfeatures=50)



        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        '''r, c = gray.shape
        s_aux = cv2.pyrDown(gray, dstsize=(c // 2, r // 2))
        s_aux = cv2.pyrUp(s_aux, dstsize=(c, r))
        gray = cv2.subtract(gray, s_aux)'''

        scale_percent = 50  # percent of original size
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        if gray.shape[0] > 500 or gray.shape[1] > 500:
            gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

        kp = sift.detect(gray, None)
        #print(kp)
        #print(len(kp))
        '''img = cv2.drawKeypoints(gray, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('sift_keypoints.jpg', img)
        cv2.waitKey(0)'''

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

    @staticmethod
    def hog2(img):
        cv2.waitKey(0)
        img = cv2.resize(img,(240,240))
        '''cv2.imshow("resized imag", img)
        cv2.waitKey(0)'''
        winSize = (240, 240)
        blockSize = (40, 40)
        blockStride = (5, 5)
        cellSize = (40, 40)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradients = True

        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)
        h = hog.compute(img)*255

        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees )
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        '''print(h)
        print(len(h))
        print()
        print(h[500])
        cv2.imshow("imag", img)
        cv2.waitKey(0)'''
        return h

    @staticmethod
    def hog3(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (120, 120))
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(10, 10),
                            cells_per_block=(2, 2), visualize=True, multichannel=False)
        fd= fd*255
        #print(fd)
        #print(len(fd))
        '''fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()'''
        return fd

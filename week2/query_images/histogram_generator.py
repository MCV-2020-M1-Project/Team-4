import cv2 as cv
import numpy as np


class HistogramGenerator(object):

    @staticmethod
    def create_hists(image):
        """
        This function load an image, filtered with some basic operations and calculate some specific histograms
        :param image: image array
        :return: array with histograms concatenated
        """
        img = image.copy()
        height, width, dimensions = img.shape
        if height > 250:
            factor = height // 250
            img = cv.resize(img, (width // factor, height // factor), interpolation=cv.INTER_AREA)
        img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
        height, width, dimensions = img.shape

        # Number of divisions
        column = 4
        row = 16
        height_cutoff = height // row
        width_cutoff = width // column
        output_array = []

        for d in range(dimensions):
            # Adaptable number of bins, to give each dimension more or less weight in the final evaluation
            if d == 0:
                bins = 16
            else:
                bins = 32
            for c in range(column):
                for r in range(row):
                    s1 = img[r * height_cutoff:(r + 1) * height_cutoff, c * width_cutoff: (c + 1) * width_cutoff, d]
                    s1_hist = np.array(cv.calcHist([s1], [0], None, [bins], [0, 256]))
                    cv.normalize(s1_hist, s1_hist)
                    output_array = np.concatenate((output_array, s1_hist), axis=None)
        return output_array

    @staticmethod
    def rgb_hist_3d(image, bins=8, mask=None):

        imgRaw = image.copy()
        hist = cv.calcHist([imgRaw], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        cv.normalize(hist, hist)
        return hist.flatten()

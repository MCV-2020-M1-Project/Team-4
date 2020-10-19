import cv2 as cv
import numpy as np

from query_images.image_utils import ImageUtils


class HistogramGenerator(object):

    @staticmethod
    def create_hists(image):
        """
        This function load an image, filtered with some basic operations and calculate some specific histograms
        :param image: image array
        :return: array with histograms concatenated
        """
        height, width, dimensions = image.shape
        image = ImageUtils.normalize_image(image)

        # Number of divisions
        column = 4
        row = 16
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
    def rgb_hist_3d(image, bins=8, mask=None):

        imgRaw = image.copy()
        hist = cv.calcHist([imgRaw], [0, 1, 2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        cv.normalize(hist, hist)
        return hist.flatten()

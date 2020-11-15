import numpy as np
import cv2


class ImageUtils(object):

    @staticmethod
    def divide_image(image, rows, cols):
        normalizeY = image.shape[0] - image.shape[0] % rows
        normalizeX = image.shape[1] - image.shape[1] % cols
        image = image[:normalizeY, :normalizeX]

        divide_columns = np.hsplit(image, cols)
        for i in range(len(divide_columns)):
            divide_columns[i] = np.vsplit(divide_columns[i], rows)

        return divide_columns

    @staticmethod
    def calc_hist(image_parts, binsSize=32):

        full_histogram = np.array([])

        for i in range(len(image_parts)):
            for j in range(len(image_parts[i])):
                histogram, bins = np.histogram(image_parts[i][j].reshape(-1), binsSize, range=[0, 256])
                normalize = np.linalg.norm(histogram)
                full_histogram = np.concatenate((full_histogram, (histogram/normalize)))

        return full_histogram

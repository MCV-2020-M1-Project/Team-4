import numpy as np
import cv2


class ImageUtils(object):

    @staticmethod
    def normalize_image(image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        height, width, dimensions = image.shape

        if height > 250:
            factor = height // 250
            image = cv2.resize(image, (width // factor, height // factor), interpolation=cv2.INTER_AREA)

        return image

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
    def calc_hist(image_parts, binsSize=32, mask=None):

        full_histogram = np.array([])

        for i in range(len(image_parts)):
            for j in range(len(image_parts[i])):
                img_slice = image_parts[i][j]
                if mask is not None:
                    img_slice = img_slice[np.where(mask[i][j]==0)]
                histogram, bins = np.histogram(img_slice.reshape(-1), binsSize, range=[0, 256])
                normalize = np.linalg.norm(histogram)
                if normalize != 0:
                    full_histogram = np.concatenate((full_histogram, (histogram/normalize)))
                else:
                    histNegative = np.zeros((binsSize)) - 0
                    full_histogram = np.concatenate((full_histogram, histNegative))

        return full_histogram

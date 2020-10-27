import cv2
import numpy as np
import pytesseract

from image_processing import ImageUtils


class DescriptorsGenerator(object):

    HISTOGRAM_CELL = 1
    TEXT = 2

    @staticmethod
    def generate_descriptor(image, method=1, cellSize=(10, 10)):
        if method == DescriptorsGenerator.HISTOGRAM_CELL:
            return DescriptorsGenerator.histogramCell(image, cellSize)
        elif method == DescriptorsGenerator.TEXT:
            return DescriptorsGenerator.getTextFromImage(image)

    @staticmethod
    def histogramCell(image, cellSize=(10,10)):

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
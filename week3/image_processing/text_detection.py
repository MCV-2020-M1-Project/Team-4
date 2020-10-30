import cv2
import numpy as np
import matplotlib.pyplot as plt

class TextDetection(object):

    @staticmethod
    def text_detection(image):
        """
        This function detects the text in the image and returns an array with coordinates of text bbox.
        input: image in BGR spacecolor.
        output: [min x, min y, max x, max y]
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert image to RGB color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert image to HSV color space
        h, s, v = cv2.split(hsv)  # split the channels of the color space in Hue, Saturation and Value
        # TextDetection.find_regions(img)
        # Open morphological transformation using a square kernel with dimensions 10x10
        kernel = np.ones((10, 10), np.uint8)
        s = cv2.GaussianBlur(s, (5, 5), 0)
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
        morph_open = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
        # Convert the image to binary
        ret, th1 = cv2.threshold(morph_open, 35, 255, cv2.THRESH_BINARY_INV)

        # Open and close morphological transformation using a rectangle kernel relative to the shape of the image
        shape = image.shape
        kernel = np.ones((shape[0] // 60, shape[1] // 4), np.uint8)
        th2 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
        # th3 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

        # Find the contours
        (contours, hierarchy) = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            th3 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)
            # th3 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
            (contours, hierarchy) = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        areaImg = shape[0] * shape[1]
        areaMinImg = areaImg * 0.01
        areaMaxImg = areaImg * 0.25
        bestContour = None
        if len(contours) > 0:
            maxArea = 0
            for c in contours:
                area = cv2.contourArea(c)
                if area > maxArea and area > areaMinImg and area < areaMaxImg:
                    maxArea = area
                    bestContour = c

        if bestContour is not None:
            rect = cv2.minAreaRect(bestContour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(rgb, [box], 0, (255, 0, 0), 2)
            x = np.array([box[0][0], box[1][0], box[2][0], box[3][0]])
            y = np.array([box[0][1], box[1][1], box[2][1], box[3][1]])
            cv2.rectangle(mask, (min(x), min(y)), ((max(x), max(y))), (255, 255, 255), -1)
            coordinates = np.array([min(x), min(y), max(x), max(y)])
        else:
            coordinates = np.zeros([4])

        # Plot the image
        """titles = ['Original with Bbox']
        images = [mask]
        for i in range(1):
            plt.subplot(1, 1, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
            plt.show()"""

        return coordinates, mask
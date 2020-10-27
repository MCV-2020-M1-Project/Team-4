import cv2
import numpy as np


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

        # Find the coordinates of the contours and draw it in the original image
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            c = contours[TextDetection.eval_contours(contours, shape[1])]
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(rgb, [box], 0, (255, 0, 0), 2)
            x = np.array([box[0][0], box[1][0], box[2][0], box[3][0]])
            y = np.array([box[0][1], box[1][1], box[2][1], box[3][1]])
            coordinates = np.array([min(x), min(y), max(x), max(y)])
            mask = np.zeros(th2.shape)
            mask[int(coordinates[1] - 5):int(coordinates[3] + 5), int(coordinates[0] - 5):int(coordinates[2] + 5)] = 255
        else:
            coordinates = np.zeros([4])
            mask = (np.ones(th2.shape) * 255).astype(np.uint8)

        # Plot the image
        # titles = ['Original with Bbox']
        # images = [rgb]
        # for i in range(1):
        # plt.subplot(1, 1, i + 1), plt.imshow(images[i], 'gray')
        # plt.title(titles[i])
        # plt.xticks([]), plt.yticks([])
        # plt.show()

        return coordinates, mask

    @staticmethod
    def eval_contours(contours, width):
        if len(contours) == 0: return 0
        if len(contours) == 1: return 0

        max_area = []
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            max_area.append(area)

        max_order = [0]
        for i in range(1, len(max_area)):
            for l in range(len(max_order) + 1):
                if l == len(max_order):
                    max_order.append(i)
                    break
                elif max_area[i] > max_area[max_order[l]]:
                    max_order.insert(l, i)
                    break

        # Get the moments
        mu = [None] * len(contours)
        for i in range(len(contours)):
            mu[i] = cv2.moments(contours[i])
        # Get the mass centers
        mc = [None] * len(contours)
        for i in range(len(contours)):
            # add 1e-5 to avoid division by zero
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

        CM_order = [0]
        for i in range(1, len(mc)):

            for l in range(len(CM_order) + 1):
                if l == len(CM_order):
                    CM_order.append(i)
                    break
                elif abs(mc[i][0] - (width / 2)) < abs(mc[CM_order[l]][0] - (width / 2)):
                    CM_order.insert(l, i)
                    break

        return CM_order[0]

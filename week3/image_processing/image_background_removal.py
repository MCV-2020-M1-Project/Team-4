import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt


class ImageBackgroundRemoval(object):

    def canny(image, show_output=False):

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]

        # Resize
        resize = imutils.resize(gray, width=gray.shape[1]//2)
        ratio = image.shape[0] / resize.shape[0]

        # Apply gaussian and edges
        gaussiana = cv2.GaussianBlur(resize, (3, 3), 1.25)
        edges = cv2.Canny(gaussiana, 50, 200)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)),iterations=1)

        # Contours detector
        (contours, hierarchy) = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        definitive_contours = []
        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1:
               definitive_contours.append(contours[i])

        # Generate mask
        mask = ImageBackgroundRemoval.generate_mask(image, definitive_contours, ratio)
        imgCrop = ImageBackgroundRemoval.crop_with_mask(image, mask);

        """plt.imshow(mask, 'gray');plt.show();
        plt.imshow(imgCrop[0]);plt.show()
        if(len(imgCrop) > 1):
            plt.imshow(imgCrop[1]);plt.show()"""

        return imgCrop

    @staticmethod
    def generate_mask(image, contours, ratio):

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # Detect the biggest contour and draw this in the mask image
        c1 = c2 = None
        if len(contours) > 0:
            array_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            c1 = ImageBackgroundRemoval.normalize_countour(array_sorted[0], ratio)

            if len(array_sorted) > 1 and cv2.contourArea(array_sorted[1]) > 4000:
                c2 = ImageBackgroundRemoval.normalize_countour(array_sorted[1], ratio)
                cv2.drawContours(mask, [c1, c2], -1, (255, 0, 0), -1)
            else:
                cv2.drawContours(mask, [c1], -1, (255, 0, 0), -1)

        return mask

    @staticmethod
    def normalize_countour(contours, ratio):
        contours = contours.astype("float")
        contours *= ratio
        contours = contours.astype("int")
        return contours

    @staticmethod
    def crop_with_mask(image, mask):

        images = []
        (contours, hierarchy) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        definitive_contours = []
        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1:
               definitive_contours.append(contours[i])

        for c in definitive_contours:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x = np.array([box[0][0], box[1][0], box[2][0], box[3][0]])
            y = np.array([box[0][1], box[1][1], box[2][1], box[3][1]])

            cropped_img = image[min(y):max(y), min(x):max(x)]

            images.append(cropped_img)

        return images
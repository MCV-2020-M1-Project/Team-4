import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import skimage

from image_processing import ImageNoise


class ImageBackgroundRemoval(object):

    @staticmethod
    def canny(image, show_output=False):

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]

        # Resize
        resize = imutils.resize(gray, width=gray.shape[1] // 2)
        ratio = image.shape[0] / resize.shape[0]

        # Apply gaussian and edges
        resize = ImageNoise.remove_noise(resize, ImageNoise.MEDIAN, 7)
        # gaussiana = cv2.GaussianBlur(resize, (1, 1), 1.25)
        edges = cv2.Canny(resize, 50, 125)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                 iterations=3)
        plt.imshow(edges, 'gray')
        plt.show()

        # Contours detector
        (contours, hierarchy) = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        definitive_contours = []
        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1 and cv2.contourArea(contours[i]) > gray.shape[1] * gray.shape[0] * 0.018:
                definitive_contours.append(contours[i])

        # Generate mask
        mask = ImageBackgroundRemoval.generate_mask(image, definitive_contours, ratio)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                iterations=1)

        imgCrop = ImageBackgroundRemoval.crop_with_mask(image, mask)

        for img in imgCrop:
            plt.imshow(img)
            plt.show()

        return imgCrop

    @staticmethod
    def generate_mask(image, contours, ratio):

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # Detect the biggest contour and draw this in the mask image

        array_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        for i in range(len(array_sorted)):
            c1 = ImageBackgroundRemoval.normalize_countour(array_sorted[i], ratio)
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
        (contours, hierarchy) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = ImageBackgroundRemoval.sort_contours(contours)[0]

        for c in contours:
            subMask = np.zeros(image.shape[:-1])
            cv2.drawContours(subMask, [c], -1, 255, -1)
            out = np.zeros_like(image)  # Extract out the object and place into output image
            out[subMask == 255] = image[subMask == 255]

            (y, x) = np.where(subMask == 255)
            (topy, topx) = (np.min(y), np.min(x))
            (bottomy, bottomx) = (np.max(y), np.max(x))
            out = image[topy:bottomy + 1, topx:bottomx + 1]

            # cropped_img = image[min(y):max(y), min(x):max(x)]

            images.append(out)

        return images

    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

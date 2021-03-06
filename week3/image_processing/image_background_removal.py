import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

from image_processing import ImageNoise


class ImageBackgroundRemoval(object):

    def canny(image, show_output=False):

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        gray = ImageNoise.remove_noise(gray, ImageNoise.MEDIAN, 9)
        th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        #plt.imshow(gray, 'gray');plt.show();

        # Resize
        resize = imutils.resize(gray, width=gray.shape[1]//2)
        ratio = image.shape[0] / resize.shape[0]

        # Apply gaussian and edges
        gaussiana = cv2.GaussianBlur(resize, (1, 1), 1.25)
        edges = cv2.Canny(gaussiana, 50, 200)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)),iterations=2)

        # Contours detector
        (contours, hierarchy) = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        definitive_contours = []
        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1 and cv2.contourArea(contours[i]) > gray.shape[1]*2:
               definitive_contours.append(contours[i])

        # Generate mask
        mask = ImageBackgroundRemoval.generate_mask(image, definitive_contours, ratio)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                 iterations=2)
        imgCrop = ImageBackgroundRemoval.crop_with_mask(image, mask);

        #plt.imshow(mask, 'gray');plt.show();
        #plt.imshow(imgCrop[0]);plt.show()
        #if(len(imgCrop) > 1):
        #    plt.imshow(imgCrop[1]);plt.show()

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
        (contours, hierarchy) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = ImageBackgroundRemoval.sort_contours(contours)[0]

        for c in contours:
            """rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x = np.array([box[0][0], box[1][0], box[2][0], box[3][0]])
            y = np.array([box[0][1], box[1][1], box[2][1], box[3][1]])"""
            subMask = np.zeros(image.shape[:-1])
            cv2.drawContours(subMask, [c], -1, 255, -1)
            out = np.zeros_like(image)  # Extract out the object and place into output image
            out[subMask == 255] = image[subMask == 255]

            (y, x) = np.where(subMask == 255)
            (topy, topx) = (np.min(y), np.min(x))
            (bottomy, bottomx) = (np.max(y), np.max(x))
            out = image[topy:bottomy + 1, topx:bottomx + 1]

            #cropped_img = image[min(y):max(y), min(x):max(x)]

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

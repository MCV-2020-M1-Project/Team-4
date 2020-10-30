import cv2
import imutils
import numpy as np

class ImageBackgroundRemoval(object):

    def method1(image, show_output=False):

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize
        resize = imutils.resize(gray, width=300)
        ratio = image.shape[0] / resize.shape[0]

        # Apply gaussian and edges
        gaussiana = cv2.GaussianBlur(resize, (3, 3), 0)
        edges = cv2.Canny(gaussiana, 50, 200)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

        # Contours detector
        (contours, hierarchy) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Generate mask
        mask = ImageBackgroundRemoval.generate_mask(image, contours, ratio)

        if show_output:
            cv2.imshow('GAU', gaussiana)
            cv2.imshow('Edges', edges)
            cv2.imshow('Image', gray)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return mask

    @staticmethod
    def generate_mask(image, contours, ratio):

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Detect the biggest contour and draw this in the mask image
        if len(contours) > 0:
            array_sorted = sorted(contours, key=cv2.contourArea)
            c1 = ImageBackgroundRemoval.normalize_countour(array_sorted[0], ratio)

            if len(array_sorted) > 1:
                c2 = ImageBackgroundRemoval.normalize_countour(array_sorted[1], ratio)
                cv2.drawContours(mask, [c1, c2], -1, (255, 0, 0), -1)
            else:
                cv2.drawContours(mask, [c1], -1, (255, 0, 0), -1)

        return mask
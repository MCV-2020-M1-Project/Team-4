import cv2
import numpy as np
import imutils


# Class to remove backgrounds and evaluate masks
class BackgroundRemove(object):

    EDGES = 1
    MORPH = 2
    THRES = 3

    # Method 1 using Edge detector
    @staticmethod
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
        mask = BackgroundRemove.generate_mask(image, contours, ratio)

        if show_output:
            cv2.imshow('GAU', gaussiana)
            cv2.imshow('Edges', edges)
            cv2.imshow('Image', gray)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return mask

    # Method 2 using Threshold and Morphological detector
    @staticmethod
    def method2(image, morph_type=2, show_output=False):

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize
        resize = imutils.resize(gray, width=500)
        ratio = image.shape[0] / resize.shape[0]

        # Apply Gaussian, Sharpness, Threshold and Morphological transformations
        gaussian = cv2.GaussianBlur(resize, (3, 3), 0)
        sharpness = cv2.filter2D(gaussian, cv2.CV_8U, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        threshold = cv2.threshold(sharpness, 90, 255, cv2.THRESH_BINARY_INV)[1]
        morphological = BackgroundRemove.apply_morph(threshold, morph_type)

        # Detect contours
        (contours, hierarchy) = cv2.findContours(morphological, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Generate mask
        mask = BackgroundRemove.generate_mask(image, contours, ratio)

        if show_output:
            cv2.imshow('Image', resize)
            cv2.imshow('Image Tresh', morphological)
            cv2.imshow('Mask', mask)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return mask

    # Method3 Color treshold
    @staticmethod
    def method3(image, show_output=False):

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)[1]
        (contours, hierarchy) = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = BackgroundRemove.generate_mask(image, contours, 1)

        if show_output:
            cv2.imshow('GAU', gray)
            cv2.imshow('Treshold', threshold)
            cv2.imshow('Mask', mask)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return mask

    # Apply to a image different morphological transformations and returns this one
    @staticmethod
    def apply_morph(image, morph_type):

        # Method1: Two closes of different size
        if morph_type == 1:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                     iterations=1)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
                                     iterations=3)

        # Method2: One close in x axis
        if morph_type == 2:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=np.uint8), iterations=3)

        # Method3: One close with CROSS structure and 15 iterations
        if morph_type == 3:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5), dtype=np.uint8),
                                     iterations=15)

        return image

    @staticmethod
    def generate_mask(image, contours, ratio):

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Detect the biggest contour and draw this in the mask image
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(mask, [c], -1, (255, 0, 0), -1)

        return mask

    @staticmethod
    def crop_with_mask(image, mask):
        rect = cv2.boundingRect(mask)
        cropped_img = image[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        return cropped_img

    # Main method to remove a query_images, returns the image cropped and the mask
    @staticmethod
    def remove_background(image, method=1, show_output=False):

        # Apply method to generate mask
        if method == BackgroundRemove.EDGES:# Edges
            mask = BackgroundRemove.method1(image, show_output)
        elif method == BackgroundRemove.MORPH:# Morphological
            mask = BackgroundRemove.method2(image, 2, show_output) #
        elif method == BackgroundRemove.THRES:# Threshold
            mask = BackgroundRemove.method3(image, show_output)

        # Generate image cropped
        image_crop = BackgroundRemove.crop_with_mask(image, mask)

        if show_output:
            cv2.imshow('Image Result', image_crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image_crop, mask

    # Method to evaluate the mask, returns precision, recall and f1-score indexes
    @staticmethod
    def evaluate_mask(original_mask, generated_mask):

        # Calculate True-Positive, False-Positive and False-Negative
        tp = np.sum(np.logical_and(original_mask, generated_mask))
        fp = np.sum(np.logical_and(np.logical_not(original_mask), generated_mask))
        fn = np.sum(np.logical_and(original_mask, np.logical_not(generated_mask)))

        # Calculate precision, recall and f1-score
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        return precision, recall, f1_score
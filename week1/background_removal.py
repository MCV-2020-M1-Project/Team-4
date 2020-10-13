import cv2
import numpy as np
import imutils


class BackgroundRemove(object):
    __count = 1

    @staticmethod
    def method1(image, show_output=False):

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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

    @staticmethod
    def method2(image, morph_type=2, show_output=False):

        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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
            cv2.imwrite(str(BackgroundRemove.__count) + ".jpg", gray)
            BackgroundRemove.__count = BackgroundRemove.__count + 1

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return mask

    @staticmethod
    def apply_morph(image, morph_type):

        # Test1
        if morph_type == 1:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                     iterations=1)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
                                     iterations=3)

        # Test2
        if morph_type == 2:
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ], dtype=np.uint8), iterations=3)

        return image

    @staticmethod
    def generate_mask(image, contours, ratio):

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

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

    @staticmethod
    def remove_background(image, show_output=False):

        mask = BackgroundRemove.method2(image, 2, show_output)
        image_crop = BackgroundRemove.crop_with_mask(image, mask)

        if show_output:
            cv2.imshow('Image Result', image_crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image_crop



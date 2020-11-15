import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from pytesseract import pytesseract, Output

from image_processing import ImageNoise
from query_images import Distance


class TextDetection(object):

    @staticmethod
    def opening(m, size=(45, 45)):
        kernel = np.ones(size, np.uint8)
        m = cv2.erode(m, kernel, iterations=1)
        m = cv2.dilate(m, kernel, iterations=1)
        return m

    @staticmethod
    def closing(m, size=(45, 45)):
        kernel = np.ones(size, np.uint8)
        m = cv2.dilate(m, kernel, iterations=1)
        m = cv2.erode(m, kernel, iterations=1)
        return m

    @staticmethod
    def brightText(img):
        """
        Generates the textboxes candidated based on TOPHAT morphological filter.
        Works well with bright text over dark background.
        Parameters
        ----------
        img : ndimage to process
        Returns
        -------
        mask: uint8 mask with regions of interest (possible textbox candidates)
        """
        kernel = np.ones((30, 30), np.uint8)
        img_orig = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

        TH = 150
        img_orig[(img_orig[:, :, 0] < TH) | (img_orig[:, :, 1] < TH) | (img_orig[:, :, 2] < TH)] = (0, 0, 0)

        img_orig = TextDetection.closing(img_orig, size=(1, int(img.shape[1] / 8)))

        return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)

    @staticmethod
    def darkText(img):
        """
        Generates the textboxes candidated based on BLACKHAT morphological filter.
        Works well with dark text over bright background.
        Parameters
        ----------
        img : ndimage to process
        Returns
        -------
        mask: uint8 mask with regions of interest (possible textbox candidates)
        """
        kernel = np.ones((30, 30), np.uint8)
        img_orig = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

        TH = 150
        img_orig[(img_orig[:, :, 0] < TH) | (img_orig[:, :, 1] < TH) | (img_orig[:, :, 2] < TH)] = (0, 0, 0)

        img_orig = TextDetection.closing(img_orig, size=(1, int(img.shape[1] / 8)))

        return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)

    @staticmethod
    def mixDarkLightText(img):
        kernel = np.ones((30,30), np.uint8)
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, np.ones((1, 10)), iterations=1)
        blackhat = cv2.morphologyEx(tophat, cv2.MORPH_BLACKHAT, np.ones((5, 5)), iterations=10)

        TH = 150
        blackhat[(blackhat[:, :, 0] < TH) | (blackhat[:, :, 1] < TH) | (blackhat[:, :, 2] < TH)] = (0, 0, 0)

        img_orig = TextDetection.closing(blackhat, size=(1, int(img.shape[1] / 8)))

        return (cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY) != 0).astype(np.uint8)


    def extract_biggest_connected_component(mask: np.ndarray) -> np.ndarray:
        """
        Extracts the biggest connected component from a mask (0 and 1's).
        Args:
            img: 2D array of type np.float32 representing the mask
        Returns : 2D array, mask with 1 in the biggest component and 0 outside
        """
        # extract all connected components
        num_labels, labels_im = cv2.connectedComponents(mask.astype(np.uint8))

        # we find and return only the biggest one
        max_val, max_idx = 0, -1
        for i in range(1, num_labels):
            area = np.sum(labels_im == i)
            if area > max_val:
                max_val = area
                max_idx = i

        return (labels_im == max_idx).astype(float)

    def get_textbox_score(m, p_shape):
        """
        Generates a score for how textbox-ish a mask connected component is.
        Parameters
        ----------
        m : mask with the textbox region with 1's
        p_shape : shape of the minimum bounding box enclosing the painting.
        Returns
        -------
        score: score based on size + shape
        """
        m = m.copy()

        # we generate the minimum bounding box for the extracted mask
        x, y, w, h = cv2.boundingRect(m.astype(np.uint8))

        # some upper and lower thresholding depending on its size and the painting size.
        if w < 10 or h < 10 or h > w:
            return 0
        if w >= p_shape[0] * 0.8 or h >= p_shape[1] / 4:
            return 0

        # we compute the score according to its shape and its size
        sc_shape = np.sum(m[y:y + h, x:x + w]) / (w * h)
        sc_size = (w * h) / (m.shape[0] * m.shape[1])

        final_score = (sc_shape + 50 * sc_size) / 2

        return final_score

    @staticmethod
    def text_detection3(image):

        dark = TextDetection.darkText(image);
        bright = TextDetection.brightText(image)
        mix = TextDetection.mixDarkLightText(image)

        darkC = TextDetection.extract_biggest_connected_component(dark)
        bC = TextDetection.extract_biggest_connected_component(bright)
        mixC = TextDetection.extract_biggest_connected_component(mix)

        scoreDark = TextDetection.get_textbox_score(darkC, dark.shape)
        scoreLight = TextDetection.get_textbox_score(bC, dark.shape)
        scoreMix = TextDetection.get_textbox_score(mixC, mix.shape)
        best = None

        if scoreDark < scoreLight:
            best = darkC
        else:
            best = bC
        #plt.imshow(best)
        #plt.show()
        mask = np.zeros(best.shape)
        contours, _ = cv2.findContours(best.astype(np.uint8), 1, 1)  # not copying here will throw an error
        rect = cv2.minAreaRect(contours[0])  # basically you can feed this rect into your classifier
        (x, y), (w, h), a = rect  # a - angle

        box = cv2.boxPoints(rect)
        box = np.int0(box)  # turn into ints
        rect2 = cv2.drawContours(mask, [box], 0, 255, 10) # Show the rectangle
        rect2 = cv2.drawContours(mask, [box], 0, 255, -1)

        #plt.imshow(rect2, 'gray')
        #plt.show()

        return rect2

    @staticmethod
    def text_detection2(image):

        blurred = cv2.GaussianBlur(image, (3, 3), 1)
        laplacian = cv2.Laplacian(image,cv2.CV_16S, ksize= 3)
        median = cv2.medianBlur(image,ksize=3)
        # esto nos puede dar problemas para calcular los vertices
        resize = imutils.resize(median, width=image.shape[1]//2)

        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

        imgGeneral = TextDetection.text_detect_general(gray)
        imgBright = TextDetection.text_detect_bright(gray)


        areaImg = image.shape[0] * image.shape[1]
        ratio = image.shape[1] / resize.shape[1]
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(imgGeneral)

        bestArea = 0
        bestStats = None
        for label in range(retval):
            actualStats = TextDetection.normalize_countour(stats[label], ratio)
            x, y, w, h, area = actualStats
            if areaImg * 0.001 < area < areaImg * 0.3 and w > h * 2 and bestArea < area and image.shape[1] * 0.3 < centroids[label,1] < image.shape[1] * 0.7:
                bestArea = area
                bestStats = actualStats

        retval1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(imgBright)

        for label in range(retval1):
            actualStats = TextDetection.normalize_countour(stats1[label], ratio)
            x, y, w, h, area = actualStats
            if areaImg * 0.001 < area < areaImg * 0.3 and w > h * 2 and bestArea < area and image.shape[0] * 0.45 < centroids1[label,0] < image.shape[0] * 0.65:
                bestArea = area
                bestStats = actualStats

        if bestStats is not None:
            x, y, w, h, area = bestStats
            cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), -1)
            #plt.imshow(image, 'gray')
            #plt.show()
        else:
            return image

        return image

    @staticmethod
    def text_detect_bright(img):
        kernel = np.ones((10, 10), np.uint8)
        kernel1, kernel2 = img.shape[0] // 20, img.shape[1] // 20
        img_TH = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, np.ones((kernel1, kernel2), np.uint8), iterations=2)
        # Give extra margin

        img_TH = img_TH[kernel1:img.shape[0] - kernel1, kernel2:img.shape[1] - kernel2]

        # Threshold adaptative 90%

        TH = int(np.max(img_TH) * 0.70)

        img_TH[(img_TH[:, :] < TH)] = 0

        kernel1, kernel2 = int(img.shape[0] // 15), int(img.shape[1] // 2)
        if kernel1 % 2 == 0:
            kernel1 += 1
        if kernel2 % 2 == 0:
            kernel2 += 1
        img_Thresh = cv2.morphologyEx(img_TH, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        img_Thresh = cv2.morphologyEx(img_Thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        img_TH = cv2.morphologyEx(img_Thresh, cv2.MORPH_TOPHAT, np.ones((kernel1, kernel2), np.uint8), iterations=2)
        img_Thresh = cv2.morphologyEx(img_Thresh, cv2.MORPH_CLOSE, np.ones((kernel1, kernel2), np.uint8), borderValue=0)
        return img_Thresh

    @staticmethod
    def text_detect_general(img):
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, np.ones((1, 10)), iterations=1)
        blackhat = cv2.morphologyEx(tophat, cv2.MORPH_BLACKHAT, np.ones((5, 5)), iterations=10)

        thres = cv2.threshold(blackhat, 100, 255, cv2.THRESH_BINARY)[1]

        open = cv2.morphologyEx(thres, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=1)
        dilate = cv2.morphologyEx(open, cv2.MORPH_DILATE, np.ones((10, 100)), iterations=1)
        return dilate

    @staticmethod
    def normalize_countour(contour, ratio):
        contour = contour.astype("float")
        contour *= ratio
        contour = contour.astype("int")
        return contour

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
        #s = cv2.GaussianBlur(s, (5, 5), 0)
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

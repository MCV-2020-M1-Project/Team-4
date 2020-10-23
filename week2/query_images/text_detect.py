import cv2
import numpy as np
from query_images import BackgroundRemove, HistogramDistance
from matplotlib import pyplot as plt

class TextDetect(object):

    @staticmethod
    def correlation(image, template):
        res = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        w, h = template.shape[::-1]
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        return (top_left, bottom_right)

    @staticmethod
    def method_morph(image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        morhp1 = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

        morhp1 = cv2.morphologyEx(morhp1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)),
                                  iterations=8)

        morhp1 = cv2.morphologyEx(morhp1, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)),
                                  iterations=1)

        (contours, hierarchy) = cv2.findContours(morhp1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            for c in contours:
                h = c[1][0][0] - c[0][0][0]
                w = c[1][0][1] - c[0][0][1]
                area = cv2.contourArea(c)
                print(area)
                #if area > 60000 and area < 120000:
                cv2.drawContours(image_gray, [c], -1, (128, 0, 0), -1)

        return image_gray;

        treshMarco = cv2.threshold(image_gray, 10, 255, cv2.THRESH_BINARY)[1]
        treshTexto = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)[1]
        treshTextoWhite = cv2.threshold(image_gray, 40, 255, cv2.THRESH_BINARY_INV)[1]

        morhp1 = cv2.morphologyEx(treshMarco, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                 iterations=6)
        morhp1 = cv2.morphologyEx(morhp1, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                  iterations=4)
        morhp1 = cv2.morphologyEx(morhp1, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                  iterations=50)
        morhp1 = cv2.morphologyEx(morhp1, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                  iterations=20)

        treshConjunto = ((treshTexto - morhp1) * 255).astype(np.uint8)
        morhp2 = cv2.morphologyEx(treshConjunto, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (8, 4)),
                                  iterations=1)
        morhp2 = cv2.morphologyEx(morhp2, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                                  iterations=10)
        morhp2 = cv2.morphologyEx(morhp2, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5)),
                                  iterations=5)
        morhp2 = cv2.morphologyEx(morhp2, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 4)),
                                  iterations=10)

        #morhp2 = cv2.threshold(morhp2, 125, 255, cv2.THRESH_BINARY)[1]


        (contours, hierarchy) = cv2.findContours(morhp2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(image_gray, [c], -1, (128, 0, 0), -1)

        return treshConjunto

    @staticmethod
    def method_corr(image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        w, h = image_gray.shape[::-1]
        if h > 500:
            factor = h // 500
            image_gray = cv2.resize(image_gray, (w // factor, h // factor), interpolation=cv2.INTER_AREA)

        w, h = image_gray.shape[::-1]

        print(h, w)

        bestDistance = 999999999
        bestTop, bestRight = None, None
        for hT in range(100, h, 10):
            for wT in range(100, w, 10):
                if wT/hT > 5:
                    print(hT, wT)
                    template = np.zeros((hT, wT), dtype=np.uint8)
                    template[:, :] = 255

                    top_left, bottom_right = TextDetect.correlation(image_gray, template)
                    cropped = image_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    hist = np.histogram(cropped.reshape(-1), 255, range=[0, 256])[0]
                    histTmp = np.histogram(template, 255, range=[0, 256])[0]
                    distance = HistogramDistance.euclidean(hist, histTmp)

                    if(distance < bestDistance):
                        bestDistance = distance
                        bestTop = top_left
                        bestRight = bottom_right

        print(bestDistance)
        cv2.rectangle(image_gray, bestTop, bestRight, 255, 2)

        return image_gray

    @staticmethod
    def method_convolve(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        v = hsv[:, :, 2]
        vDelta = np.abs(np.roll(v, 1) - v)

        s = hsv[:, :, 1]
        sDelta = np.abs(np.roll(s, 1) - s)

        return (vDelta * (1 - sDelta/100)).astype(np.uint8)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        out = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)),
                               iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)),
                               iterations=10)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                               iterations=3)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                               iterations=4)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                               iterations=6)



        out = cv2.threshold(out, 30, 255, cv2.THRESH_BINARY)[1]

        return out


        return out


    @staticmethod
    def test(img):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        kernel = np.ones((10, 10), np.uint8)
        morph_open = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
        ret, th1 = cv2.threshold(morph_open, 40, 255, cv2.THRESH_BINARY_INV)

        shape = img.shape
        x = shape[0] // 50
        y = shape[1] // 5
        kernel = np.ones((x, y), np.uint8)
        th2 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
        th3 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

        (contours, hierarchy) = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        areaImg = shape[0]*shape[1]
        areaMinImg = areaImg * 0.01
        areaMaxImg = areaImg * 0.24
        print(areaMinImg, areaMaxImg)
        if len(contours) > 0:
            maxArea = 0
            cR = None
            for c in contours:
                area = cv2.contourArea(c)
                print(area)
                if area > maxArea and area > areaMinImg and area < areaMaxImg:
                    maxArea = area
                    cR = c

            print("Best "+str(maxArea))
            if cR is not None:
                cv2.drawContours(img, [cR], -1, (128, 0, 0), -1)

        titles = ['Original', 'With Bounding Box', 'Dilate op.', 'Open op.']
        images = [img_RGB, img, th2, th3]
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    @staticmethod
    def best_method(img):

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert image to HSV color space
        h, s, v = cv2.split(hsv)  # split the channels of the color space in Hue, Saturation and Value
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

        # Open morphological transformation using a square kernel with dimensions 10x10
        kernel = np.ones((15, 15), np.uint8)
        morph_open = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
        # Convert the image to binary
        ret, th1 = cv2.threshold(morph_open, 20, 255, cv2.THRESH_BINARY_INV)

        # Open and close morphological transformation using a rectangle kernel relative to the shape of the image
        shape = img.shape
        kernel = np.ones((shape[0] // 60, shape[1] // 4), np.uint8)
        th2 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
        th3 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
        # Find the contours
        (contours, hierarchy) = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        areaImg = shape[0] * shape[1]
        areaMinImg = areaImg * 0.01
        areaMaxImg = areaImg * 0.24
        bestContour = None
        if len(contours) > 0:
            maxArea = 0
            for c in contours:
                area = cv2.contourArea(c)
                print(area)
                if area > maxArea and area > areaMinImg and area < areaMaxImg:
                    maxArea = area
                    bestContour = c

            print("Best "+str(maxArea))

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
        '''
        titles = ['Original', 'With Bounding Box', 'Dilate op.', 'Open op.']
        images = [rgb, img, mask, th3]
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
        '''
        return img, mask, coordinates
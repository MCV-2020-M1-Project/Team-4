import pickle

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import math

# Function correct_orientation
#from image_processing import ImageBackgroundRemoval
#from query_images import Distance

class Rotation(object):
    @staticmethod
    def correct_orientation(image):
        resize = imutils.resize(image, width=250)
        imgCpy = image.copy()

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(resize, 5 )

        edge_image = cv2.Canny(img, 40, 150)
        #plt.imshow(edge_image)
        #plt.show()

        th = 500
        eps = 20
        minLines = 10
        lines = None
        resH = []
        while (lines is None or len(resH) < minLines) and th > 0:
            lines = cv2.HoughLines(edge_image, 1, np.pi / 180, th, srn=0, stn=0)
            v, resH = Rotation.get_lines(lines)
            if lines is None or len(resH) < minLines:
                th -= eps

        # print("Iterations " + str(it))
        deg = []

        if lines is not None:
            for i in range(0, len(resH)):
                rho = resH[i][0][0]
                theta = resH[i][0][1]
                degrees = math.degrees(theta)
                a = math.cos(theta)
                b = math.sin(theta)
                deg.append(degrees)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 4000 * (-b)), int(y0 + 4000 * (a)))
                pt2 = (int(x0 - 4000 * (-b)), int(y0 - 4000 * (a)))
                cv2.line(imgCpy, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        rot = np.mean(deg)
        if not math.isnan(rot):
            return imutils.rotate_bound(image, 90 - rot), 90 - rot

        return image, 0

    @staticmethod
    def get_lines(lines):
        lines_v = []
        lines_h = []

        if lines is not None:
            for lineI in range(0, len(lines)):
                theta = lines[lineI][0][1]

                if theta > np.pi / 180 * 170 or theta < np.pi / 180 * 10:
                    lines_v.append(lines[lineI])

                if np.pi / 180 * 50 < theta < np.pi / 180 * 120:
                    lines_h.append(lines[lineI])

        return lines_v, lines_h

    @staticmethod
    def fnTry(self):
        # %%
        file = open('qsd1_w5/frames.pkl', 'rb')
        items = pickle.load(file)
        exp = np.array([item[0][0] for item in items])
        res = []

        for i in range(4, 30):
            image = cv2.imread('qsd1_w5/{:05d}.jpg'.format(i))
            imgRes, rad = Rotation.correct_orientation(image)
            res.append(rad)
            imageFinal = cv2.cvtColor(imgRes, cv2.COLOR_BGR2RGB)
            plt.imshow(imageFinal)
            plt.show()

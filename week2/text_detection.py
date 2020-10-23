import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt


def text_detection(img):
    """
    This function detects the text in the image and returns an array with coordinates of text bbox.
        input: image in BGR spacecolor.
        output: [tlx1, tly1, brx1, bry1] where t = top, b = bottom, l = left, r = right
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert image to HSV color space
    h, s, v = cv2.split(hsv)  # split the channels of the color space in Hue, Saturation and Value

    # Open morphological transformation using a square kernel with dimensions 10x10
    kernel = np.ones((10, 10), np.uint8)
    morph_open = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
    # Convert the image to binary
    ret, th1 = cv2.threshold(morph_open, 30, 255, cv2.THRESH_BINARY_INV)

    # Open and close morphological transformation using a rectangle kernel relative to the shape of the image
    shape = img.shape
    kernel = np.ones((shape[0] // 50, shape[1] // 5), np.uint8)
    th2 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
    th3 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
    # Find the contours
    (contours, hierarchy) = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Find the coordinates of the contours and draw it in the original image
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(rgb, [box], 0, (255, 0, 0), 2)
        coordinates = np.concatenate([box[0], box[2]])
    else:
        coordinates = np.zeros([4])
    '''
    #Plot the image
    titles = ['Original with Bbox']
    images = [rgb]
    for i in range(1):
        plt.subplot(1, 1, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    '''
    return coordinates


tboxesfile = open('qsd1_w2/text_boxes.pkl', 'rb')
tboxes = pickle.load(tboxesfile)
tboxesfile.close()
original_tboxes = np.empty([30, 4], dtype=int)
for i in range(len(tboxes)):
    original_tboxes[i] = np.concatenate([tboxes[i][0][1], tboxes[i][0][3]])
tboxes_list = list(original_tboxes)

query_tboxes = np.empty([30, 4], dtype=int)
for i in range(30):
    img = cv2.imread('qsd1_w2/{:05d}.jpg'.format(i))
    query_tboxes[i] = text_detection(img)
query_list = list(query_tboxes)

print('Original coordinates: ', tboxes_list)
print('Finded coordinates: ', query_list)

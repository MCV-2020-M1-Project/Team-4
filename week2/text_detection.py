import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt


def text_detection(img):
    """
    This function detects the text in the image and returns an array with coordinates of text bbox.
        input: image in BGR spacecolor.
        output: [min x, min y, max x, max y]
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
        x = np.array([box[0][0],box[1][0],box[2][0],box[3][0]])
        y = np.array([box[0][1],box[1][1],box[2][1],box[3][1]])
        coordinates = np.array([min(x),min(y),max(x),max(y)])
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


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


tboxesfile = open('qsd1_w2/text_boxes.pkl', 'rb')
tboxes = pickle.load(tboxesfile)
tboxesfile.close()
original_tboxes = np.empty([30, 4], dtype=int)
for i in range(len(tboxes)):
    original_tboxes[i] = np.concatenate([tboxes[i][0][0], tboxes[i][0][2]])
tboxes_list = list(original_tboxes)

query_tboxes = np.empty([30, 4], dtype=int)
metric_IoU = np.empty([30], dtype=float)
for i in range(30):
    img = cv2.imread('qsd1_w2/{:05d}.jpg'.format(i))
    query_tboxes[i] = text_detection(img)
    metric_IoU[i] = bb_intersection_over_union(query_tboxes[i], original_tboxes[i])
query_list = list(query_tboxes)

metric_iou_mean = np.mean(metric_IoU)
print('Original text positions: ', tboxes_list)
print('Finded text positions: ', query_list)
print('metric_IoU average: ',metric_iou_mean)

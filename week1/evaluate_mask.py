import cv2
from week1.background_removal import BackgroundRemove

N_IMAGES = 30


def evaluate(method):
    precision = 0
    recall = 0
    f1_score = 0

    for i in range(0, N_IMAGES):
        img = cv2.imread("qsd2_w1/{:05d}.jpg".format(i))
        mask_groundtruth = cv2.imread("qsd2_w1/{:05d}.png".format(i), cv2.IMREAD_GRAYSCALE)
        img, mask = BackgroundRemove.remove_background(img, method)

        (p, r, f1) = BackgroundRemove.evaluate_mask(mask_groundtruth, mask)
        precision = precision + p
        recall = recall + r
        f1_score = f1_score + f1

    precision = precision / N_IMAGES
    recall = recall / N_IMAGES
    f1_score = f1_score / N_IMAGES
    print(precision, recall, f1_score)


if __name__ == "__main__":
    evaluate(BackgroundRemove.EDGES)
    evaluate(BackgroundRemove.MORPH)
    evaluate(BackgroundRemove.TRESH)

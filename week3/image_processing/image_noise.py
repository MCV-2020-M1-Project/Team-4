import cv2

class ImageNoise(object):

    GAUSSIAN = 1
    AVERAGE = 2
    MEDIAN = 3

    @staticmethod
    def remove_noise(img, method=1, size=5):
        if method == ImageNoise.GAUSSIAN:
            return ImageNoise.remove_noise_gaussian(img)
        elif method == ImageNoise.AVERAGE:
            return ImageNoise.remove_noise_average(img)
        elif method == ImageNoise.MEDIAN:
            return ImageNoise.remove_noise_median(img, size)

        return img

    @staticmethod
    def remove_noise_gaussian(img):
        return cv2.GaussianBlur(img, (25, 25), 0.5)

    @staticmethod
    def remove_noise_average(img):
        return cv2.blur(img, (3, 3))

    @staticmethod
    def remove_noise_median(img, size):
        return cv2.medianBlur(img, size)

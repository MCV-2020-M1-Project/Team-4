import cv2

class ImageNoise(object):

    GAUSSIAN = 1
    AVERAGE = 2

    @staticmethod
    def remove_noise(img, method=1):
        if method == ImageNoise.GAUSSIAN:
            return ImageNoise.remove_noise_gaussian(img)
        elif method == ImageNoise.AVERAGE:
            return ImageNoise.remove_noise_average(img)

        return img

    @staticmethod
    def remove_noise_gaussian(img):
        return cv2.GaussianBlur(img, (3, 3), 0)

    @staticmethod
    def remove_noise_average(img):
        return cv2.blur(img, (3, 3))

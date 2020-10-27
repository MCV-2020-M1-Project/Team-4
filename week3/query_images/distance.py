import numpy as np


class Distance(object):

    @staticmethod
    def euclidean(h1, h2):  # Euclidean distance
        return np.linalg.norm(h1 - h2)

    @staticmethod
    def x2distance(h1, h2):  # x^2 distance
        h1 = h1 + 0.0000000001
        return np.sum(np.divide(np.power((h1 - h2), 2), (h1 + h2)))
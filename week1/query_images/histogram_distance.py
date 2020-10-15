import numpy as np


class HistogramDistance(object):

    @staticmethod
    def euclidean2(h1, h2):  # Euclidean distance
        return np.linalg.norm(h1 - h2)

    @staticmethod
    def x2distance(h1, h2):  # x^2 distance
        return np.sum(np.power((h1 - h2), 2) / (h1 + h2))
import math


class HistogramDistance(object):

    @staticmethod
    def euclidean(h1, h2):  # Euclidean distance
        result = 0
        for k in range(len(h1)):
            dif = (h1[k] - h2[k]) ** 2
            result += dif

        return math.sqrt(result)

    @staticmethod
    def x2distance(h1, h2):  # x^2 distance
        result = 0
        l = len(h1)
        for k in range(l):
            if k == l / 2:
                if (l * 0.12) / 2 < result:
                    result = l
                    return result
            h11 = h1[k]
            h22 = h2[k]
            if (h11 + h22) == 0:
                dif = 0
            else:
                dif = ((h11 - h22) ** 2) / (h11 + h22)
            result += dif

        return result
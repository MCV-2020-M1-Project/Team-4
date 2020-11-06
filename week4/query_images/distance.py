import numpy as np
import cv2


class Distance(object):

    @staticmethod
    def euclidean(h1, h2):  # Euclidean distance
        return np.linalg.norm(h1 - h2)

    @staticmethod
    def x2distance(h1, h2):  # x^2 distance
        h1 = h1 + 0.0000000001
        return np.sum(np.divide(np.power((h1 - h2), 2), (h1 + h2)))

    @staticmethod
    def matches(des1, des2):  # x^2 distance

        count = 0
        try:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.5 * n.distance:
                    good.append([m])
                    count += 1
        except:
            pass

        return count

    @staticmethod
    def levenshtein(str1, str2):  # x^2 distance
        distances = np.zeros((len(str1) + 1, len(str2) + 1))

        for t1 in range(len(str1) + 1):
            distances[t1][0] = t1

        for t2 in range(len(str2) + 1):
            distances[0][t2] = t2

        a = 0
        b = 0
        c = 0

        for t1 in range(1, len(str1) + 1):
            for t2 in range(1, len(str2) + 1):
                if (str1[t1 - 1] == str2[t2 - 1]):
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]

                    if (a <= b and a <= c):
                        distances[t1][t2] = a + 1
                    elif (b <= a and b <= c):
                        distances[t1][t2] = b + 1
                    else:
                        distances[t1][t2] = c + 1

        return distances[len(str1)][len(str2)]

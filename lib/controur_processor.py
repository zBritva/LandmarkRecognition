# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np


class ContourProcessor:
    def __init__(self):
        print 'do somethig'

    # cm
    focus_distance = 2
    # cm
    mark_size = 18.5

    def __distance__(self, point1, point2):
        return np.sqrt(np.power(point1[0] - point2[0], 2) + np.power(point1[1] - point2[1], 2))

    def __middlePoint__(self, point1, point2, ratio=1):
        x = (point1[0] + ratio * point2[0]) / (1 + ratio)
        y = (point1[1] + ratio * point2[1]) / (1 + ratio)

        return np.int0([x, y])

    def createLineIterator(self, P1, P2, img, binary=False):

        # define local variables for readability

        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        # predefine numpy array for output based on distance between points

        # itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3))
        itbuffer.fill(np.nan)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  # vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
        elif P1Y == P2Y:  # horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
        else:  # diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX.astype(np.float32) / dY.astype(np.float32)
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
            else:
                slope = dY.astype(np.float32) / dX.astype(np.float32)
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

        # Remove points outside of image
        colX = itbuffer[:, 0]
        colY = itbuffer[:, 1]
        itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

        # itbuffer[0][2] = [0, 0, 0]

        # Get intensities from img ndarray
        # itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

        # TODO CHECK PERFORMANCE PROBLEM!!!
        result = list()
        if not binary:
            for index in range(len(itbuffer)):
                color = img[itbuffer[index][1].astype(np.uint), itbuffer[index][0].astype(np.uint)]
                result.append([itbuffer[index][1], itbuffer[index][0], color])
        else:
            for index in range(len(itbuffer)):
                color = img[itbuffer[index][1].astype(np.uint), itbuffer[index][0].astype(np.uint)]
                result.append([itbuffer[index][1], itbuffer[index][0], color])

        return result

    def rgbToGray(self, color):
        #     Y = .2126 * R^gamma + .7152 * G^gamma + .0722 * B^gamma
        gray = color[0] * .2126 + color[1] * .7152 + color[2] * .072
        return int(gray)

    def getLineAvgColor(self, iterator, binary=False):
        sum = 0

        if not binary:
            for pixel in iterator:
                sum += self.rgbToGray(pixel[2])
        else:
            for pixel in iterator:
                sum += pixel[2]

        # debug
        if len(iterator) == 0:
            raise Exception('Incorrect line')

        # if not binary:
        return int(sum * 1.0 / len(iterator))
        # else:
        #     return sum * 1.0 / len(iterator)

    def checkAvg(self, avg, accept_percentage, binary=False):
        # 2.55 - 1 percent of 255
        # if not binary:
        return int(avg * 1.0 / 2.55) > accept_percentage
        # else:
        #     return avg * 1.0 / 0.01 > accept_percentage

    def shiftContour(self, contour, x, y):
        for point in contour:
            point[0][0] -= x
            point[0][1] -= y

        return contour

    def isInCircleInside(self, contour, center, radius):
        try:
            for point in contour:
                distance = self.__distance__(center, point[0])
                if distance > radius:
                    return False
        except Exception as ex:
            print ex
            return False
        return True

    def calculateDistance(self, radius):
        # return self.focus_distance / (self.mark_size * (radius)) * 500 * 153
        return self.focus_distance + (self.focus_distance * self.mark_size / radius) * 183

    def drawMark(self, img, points, shift):
        for point in points:
            point[0] += shift[0]
            point[1] += shift[1]
            cv2.circle(img, (point[0], point[1]), 3, (128, 128, 0), 2)

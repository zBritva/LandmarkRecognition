# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np
from math import fabs
from math import sqrt


class ContourProcessor:
    def __distance__(self, point1, point2):
        return np.sqrt(np.power(point1[0] - point2[0], 2) + np.power(point1[1] - point2[1], 2))

    def __middlePoint__(self, point1, point2, ratio=1):
        x = (point1[0] + ratio * point2[0]) / (1 + ratio)
        y = (point1[1] + ratio * point2[1]) / (1 + ratio)

        return np.int0([x, y])

    def getBoxROI(self, box):

        # points of interest
        #
        # ##########
        # #        #
        # # 3#  #4 #
        # #  #  #  #
        # # 2####5 #
        # #  #  #  #
        # # 1#  #6 #
        # #        #
        # ##########
        #

        point1 = box[0]
        point2 = box[1]
        point3 = box[2]
        point4 = box[3]

        distance1 = self.__distance__(point1, point2)
        distance2 = self.__distance__(point2, point3)

        middlePoint1 = None
        middlePoint2 = None

        middlePoint1_10 = None
        middlePoint1_90 = None

        middlePoint2_10 = None
        middlePoint2_90 = None

        if distance1 > distance2:
            middlePoint1 = self.__middlePoint__(point1, point2)
            middlePoint2 = self.__middlePoint__(point3, point4)

            middlePoint1_10 = self.__middlePoint__(point1, point2, 0.1)
            middlePoint1_90 = self.__middlePoint__(point1, point2, 10)
            middlePoint2_10 = self.__middlePoint__(point3, point4, 0.1)
            middlePoint2_90 = self.__middlePoint__(point3, point4, 10)
        else:
            middlePoint1 = self.__middlePoint__(point2, point3)
            middlePoint2 = self.__middlePoint__(point4, point1)

            middlePoint1_10 = self.__middlePoint__(point2, point3, 0.1)
            middlePoint1_90 = self.__middlePoint__(point2, point3, 10)
            middlePoint2_10 = self.__middlePoint__(point4, point1, 0.1)
            middlePoint2_90 = self.__middlePoint__(point4, point1, 10)

        # find point 2 and 5
        hmark_point_2 = self.__middlePoint__(middlePoint1, middlePoint2, 0.1)
        hmark_point_5 = self.__middlePoint__(middlePoint1, middlePoint2, 10)

        # find point 1 and 6
        hmark_point_6 = self.__middlePoint__(middlePoint2_10, middlePoint1_90, 0.1)
        hmark_point_1 = self.__middlePoint__(middlePoint2_10, middlePoint1_90, 10)

        # find point 3 and 4
        hmark_point_3 = self.__middlePoint__(middlePoint1_10, middlePoint2_90, 0.1)
        hmark_point_4 = self.__middlePoint__(middlePoint1_10, middlePoint2_90, 10)

        # TODO add 7 and 8
        hmark_point_8 = self.__middlePoint__(hmark_point_2, hmark_point_6)
        hmark_point_7 = self.__middlePoint__(hmark_point_3, hmark_point_5)

        return [hmark_point_1, hmark_point_2, hmark_point_3, hmark_point_4, hmark_point_5, hmark_point_6, hmark_point_7,
                hmark_point_8]

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

        if not binary:
            return int(sum * 1.0 / len(iterator))
        else:
            return sum * 1.0 / len(iterator)

    def checkAvg(self, avg, accept_percentage, binary=False):
        # 2.55 - 1 percent of 255
        if not binary:
            return int(avg * 1.0 / 2.55) > accept_percentage
        else:
            return avg * 1.0 / 0.01 > accept_percentage

    def checkBoxROI(self, image, h_mark_points, binary=False, accept_percentage=85):
        point1 = h_mark_points[0]
        point2 = h_mark_points[1]
        point3 = h_mark_points[2]
        point4 = h_mark_points[3]
        point5 = h_mark_points[4]
        point6 = h_mark_points[5]
        point7 = h_mark_points[6]
        point8 = h_mark_points[7]

        try:

            # 1 - 3
            iterator_1_3 = self.createLineIterator(point1, point3, image, binary)
            avg_color_1_3 = self.getLineAvgColor(iterator_1_3, binary)
            if not self.checkAvg(avg_color_1_3, accept_percentage, binary):
                return False

            # 4 - 6
            iterator_4_6 = self.createLineIterator(point4, point6, image, binary)
            avg_color_4_6 = self.getLineAvgColor(iterator_4_6, binary)
            if not self.checkAvg(avg_color_4_6, accept_percentage, binary):
                return False
            # 2 - 5
            iterator_2_5 = self.createLineIterator(point2, point5, image, binary)
            avg_color_2_5 = self.getLineAvgColor(iterator_2_5, binary)
            if not self.checkAvg(avg_color_2_5, accept_percentage, binary):
                return False

            # check point 7
            check_7_point_1_1 = np.append(point7[0] - 5, point7[1])
            check_7_point_1_2 = np.append(point7[0] + 5, point7[1])
            check_7_point_2_1 = np.append(point7[0], point7[1] - 5)
            check_7_point_2_2 = np.append(point7[0], point7[1] + 5)

            # debug
            # cv2.line(image, (check_7_point_1_1[0], check_7_point_1_1[1]), (check_7_point_1_2[0], check_7_point_1_2[1]), (255, 255, 255), 3)
            # cv2.line(image, (check_7_point_1_2[0], check_7_point_1_2[1]), (check_7_point_1_2[0], check_7_point_1_2[1]), (255, 255, 255), 3)
            #
            # cv2.line(image, (check_7_point_2_1[0], check_7_point_2_1[1]), (check_7_point_2_2[0], check_7_point_2_2[1]), (255, 255, 255), 3)
            # cv2.line(image, (check_7_point_2_2[0], check_7_point_2_2[1]), (check_7_point_2_2[0], check_7_point_2_2[1]), (255, 255, 255), 3)

            iterator_71 = self.createLineIterator(check_7_point_1_1, check_7_point_1_2, image, binary)
            avg_color_71 = self.getLineAvgColor(iterator_71, binary)

            iterator_72 = self.createLineIterator(check_7_point_2_1, check_7_point_2_2, image, binary)
            avg_color_72 = self.getLineAvgColor(iterator_72, binary)

            avg_color_7 = int(avg_color_71 + avg_color_72 / 2.0)
            # 255 - because color must be black
            color = None
            if not binary:
                color = 255 - avg_color_7
            else:
                color = 1 - avg_color_7
            # if not self.checkAvg(color, accept_percentage, binary):
            #     return False

            # check point 8
            check_8_point_1_1 = np.append(point8[0] - 5, point8[1])
            check_8_point_1_2 = np.append(point8[0] + 5, point8[1])
            check_8_point_2_1 = np.append(point8[0], point8[1] - 5)
            check_8_point_2_2 = np.append(point8[0], point8[1] + 5)

            # debug
            # cv2.line(image, (check_8_point_1_1[0], check_8_point_1_1[1]), (check_8_point_1_2[0], check_8_point_1_2[1]), (255, 255, 255), 3)
            # cv2.line(image, (check_8_point_1_2[0], check_8_point_1_2[1]), (check_8_point_1_2[0], check_8_point_1_2[1]), (255, 255, 255), 3)
            #
            # cv2.line(image, (check_8_point_2_1[0], check_8_point_2_1[1]), (check_8_point_2_2[0], check_8_point_2_2[1]), (255, 255, 255), 3)
            # cv2.line(image, (check_8_point_2_2[0], check_8_point_2_2[1]), (check_8_point_2_2[0], check_8_point_2_2[1]), (255, 255, 255), 3)

            iterator_81 = self.createLineIterator(check_8_point_1_1, check_8_point_1_2, image, binary)
            avg_color_81 = self.getLineAvgColor(iterator_81, binary)

            iterator_82 = self.createLineIterator(check_8_point_2_1, check_8_point_2_2, image, binary)
            avg_color_82 = self.getLineAvgColor(iterator_82, binary)

            avg_color_8 = int(avg_color_81 + avg_color_82 / 2.0)
            # 255 - because color must be black
            color = None
            if not binary:
                color = 255 - avg_color_8
            else:
                color = 1 - avg_color_8
            # if not self.checkAvg(color, accept_percentage, binary):
            #     return False

        except Exception as ex:
            print ex
            return False

        return True

    def findHMarkhelipad(self, source_image, contours):
        for contour in contours:
            contour = cv2.approxPolyDP(contour, 2, True)

            if len(contour) > 5:  # and fabs(cv2.arcLength(contour, True)) < 20:

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)


                # cv2.drawContours(gray,[box],0,(0,0,255),2)
                # cv2.drawContours(frame, [contour], -1, (0,255,0), 3)

    def shiftContour(self, contour, x, y):
        for point in contour:
            point[0][0] -= x
            point[0][1] -= y

        return contour


    def isInCircleInside(self, contour, center, radius):
        try:
            for point in contour:
                distance = self.__distance__(center, point)
                if distance > radius:
                    return False
        except Exception as ex:
            print ex
            return False
        return True

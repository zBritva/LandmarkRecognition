# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np
from math import fabs
from math import sqrt
from contour_processor import ContourProcessor


class EMarkProcessor(ContourProcessor):
    """
    Contains methods for determining E-mark helipad
    """
    def getBoxROI(self, box):
        """
        Return interest coordinates of points for recognizing mark

        :param box: a array of box given by :func:`cv2.boxPoints`

        :return: coordinates of point
        """

        # points of interest
        #
        # ##########
        # #        #
        # # 3####4 #
        # #  #7 8  #
        # # 2####5 #
        # #  #9 10 #
        # # 1####6 #
        # #        #
        # ##########
        #

        point1 = box[0]
        point2 = box[1]
        point3 = box[2]
        point4 = box[3]

        distance1 = self.distance(point1, point2)
        distance2 = self.distance(point2, point3)

        middle_point_1 = None
        middle_point_2 = None

        middle_point_1_10 = None
        middle_point_1_90 = None

        middle_point_2_10 = None
        middle_point_2_90 = None

        if distance1 > distance2:
            middle_point_1 = self.middlePoint(point1, point2)
            middle_point_2 = self.middlePoint(point3, point4)

            middle_point_1_10 = self.middlePoint(point1, point2, 0.1)
            middle_point_1_90 = self.middlePoint(point1, point2, 10)
            middle_point_2_10 = self.middlePoint(point3, point4, 0.1)
            middle_point_2_90 = self.middlePoint(point3, point4, 10)
        else:
            middle_point_1 = self.middlePoint(point2, point3)
            middle_point_2 = self.middlePoint(point4, point1)

            middle_point_1_10 = self.middlePoint(point2, point3, 0.1)
            middle_point_1_90 = self.middlePoint(point2, point3, 10)
            middle_point_2_10 = self.middlePoint(point4, point1, 0.1)
            middle_point_2_90 = self.middlePoint(point4, point1, 10)

        # find point 2 and 5
        mark_point_2 = self.middlePoint(middle_point_1, middle_point_2, 0.1)
        mark_point_5 = self.middlePoint(middle_point_1, middle_point_2, 10)

        # find point 1 and 6
        mark_point_6 = self.middlePoint(middle_point_2_10, middle_point_1_90, 0.1)
        mark_point_1 = self.middlePoint(middle_point_2_10, middle_point_1_90, 10)

        # find point 3 and 4
        mark_point_3 = self.middlePoint(middle_point_1_10, middle_point_2_90, 0.1)
        mark_point_4 = self.middlePoint(middle_point_1_10, middle_point_2_90, 10)

        # TODO add 7 and 8
        middle_point_2_3 = self.middlePoint(mark_point_2, mark_point_3)
        middle_point_4_5 = self.middlePoint(mark_point_4, mark_point_5)

        mark_point_7 = self.middlePoint(middle_point_2_3, middle_point_4_5, 0.3)
        mark_point_8 = self.middlePoint(middle_point_4_5, middle_point_2_3, 0.3)

        # TODO add 9 and 10
        middle_point_1_2 = self.middlePoint(mark_point_1, mark_point_2)
        middle_point_5_6 = self.middlePoint(mark_point_5, mark_point_6)

        mark_point_9 = self.middlePoint(middle_point_1_2, middle_point_5_6, 0.3)
        mark_point_10 = self.middlePoint(middle_point_5_6, middle_point_1_2, 0.3)

        return [mark_point_1, mark_point_2, mark_point_3, mark_point_4, mark_point_5, mark_point_6, mark_point_7,
                mark_point_8, mark_point_9, mark_point_10]

    def checkBoxROIToHMark(self, image, e_mark_points, binary=False, accept_percentage=85):
        """
        Check coordinates given by :func:`EMarkProcessor.getBoxROI`

        :param image: binary image where was found landmark
        :param h_mark_points: a array of point from given by :func:`EMarkProcessor.getBoxROI`
        :param binary: Should be True if image is binary format
        :param accept_percentage: accept percentage when landmark will mark as recognized

        :return:
        """
        point1 = e_mark_points[0]
        point2 = e_mark_points[1]
        point3 = e_mark_points[2]
        point4 = e_mark_points[3]
        point5 = e_mark_points[4]
        point6 = e_mark_points[5]
        point7 = e_mark_points[6]
        point8 = e_mark_points[7]
        point9 = e_mark_points[8]
        point10 = e_mark_points[9]

        try:

            # 1 - 3
            iterator_1_3 = self.createLineIterator(point1, point3, image, binary)
            avg_color_1_3 = self.getLineAvgColor(iterator_1_3, binary)

            # 4 - 6
            iterator_4_6 = self.createLineIterator(point4, point6, image, binary)
            avg_color_4_6 = self.getLineAvgColor(iterator_4_6, binary)

            # calso heck that mark is not 8 number
            if not self.checkAvg(avg_color_1_3, accept_percentage) \
                    and not self.checkAvg(avg_color_4_6, accept_percentage) \
                    or self.checkAvg(avg_color_1_3, accept_percentage) \
                            and self.checkAvg(avg_color_4_6, accept_percentage):
                return False

            # 3 - 4
            iterator_3_4 = self.createLineIterator(point3, point4, image, binary)
            avg_color_3_4 = self.getLineAvgColor(iterator_3_4, binary)
            if not self.checkAvg(avg_color_3_4, accept_percentage):
                return False

            # 2 - 5
            iterator_2_5 = self.createLineIterator(point2, point5, image, binary)
            avg_color_2_5 = self.getLineAvgColor(iterator_2_5, binary)
            if not self.checkAvg(avg_color_2_5, accept_percentage):
                return False

            # 1 - 6
            iterator_1_6 = self.createLineIterator(point1, point6, image, binary)
            avg_color_1_6 = self.getLineAvgColor(iterator_1_6, binary)
            if not self.checkAvg(avg_color_1_6, accept_percentage):
                return False

            # 7 - 8 (black)
            iterator_7_8 = self.createLineIterator(point7, point8, image, binary)
            avg_color_7_8 = self.getLineAvgColor(iterator_7_8, binary)

            # color = None
            # if not binary:
            color = 255 - avg_color_7_8
            # else:
            #     color = 1 - avg_color_7_8

            if not self.checkAvg(color, accept_percentage):
                return False

            # 9 - 10 (black)
            iterator_9_10 = self.createLineIterator(point9, point10, image, binary)
            avg_color_9_10 = self.getLineAvgColor(iterator_9_10, binary)

            # color = None
            # if not binary:
            color = 255 - avg_color_9_10
            # else:
            #     color = 1 - avg_color_9_10

            if not self.checkAvg(color, accept_percentage):
                return False

        except Exception as ex:
            print ex
            return False

        return True

    def drawMarkType(self, img):
        """
        Display text with marktype E

        :param img: image for drawing

        :return: None
        """
        cv2.putText(img, 'MARK TYPE: E',
                    (0, 0 + 55), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0))

    def drawMark(self, img, e_mark_points, shift):
        """
        Draw mark on image for visual display

        :param img: image for drawing
        :param h_mark_points: point of mark given by :func:`EMarkProcessor.getBoxROI`
        :param shift: shift coordinates of mark

        .. warning:: parameter shift is obsolete

        :return: None
        """
        # super(EMarkProcessor, self).drawMark(img, e_mark_points, shift)

        point1 = e_mark_points[0]
        point2 = e_mark_points[1]
        point3 = e_mark_points[2]
        point4 = e_mark_points[3]
        point5 = e_mark_points[4]
        point6 = e_mark_points[5]
        point7 = e_mark_points[6]
        point8 = e_mark_points[7]

        cv2.line(img, (point1[0], point1[1]), (point3[0], point3[1]), (255, 0, 0))
        cv2.line(img, (point3[0], point3[1]), (point4[0], point4[1]), (255, 0, 0))
        cv2.line(img, (point1[0], point1[1]), (point6[0], point6[1]), (255, 0, 0))
        cv2.line(img, (point2[0], point2[1]), (point5[0], point5[1]), (255, 0, 0))

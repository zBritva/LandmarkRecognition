# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import numpy as np


class ContourProcessor:

    # def __init__(self, ):

    # Находит самую правую и самую нижнюю точку контура
    def __getMaxSide__(self, contour):
        maxX = contour[0][0][0]
        maxY = contour[0][0][1]

        for point in contour:
            if point[0][0] > maxX:
                maxX = point[0][0]
            if point[0][1] > maxY:
                maxY = point[0][1]

        return maxX, maxY

    # Находит самую левую и самую верхнюю точку контура
    def __getMinSide__(self, contour):
        minX = contour[0][0][0]
        minY = contour[0][0][1]

        for point in contour:
            if point[0][0] < minX:
                minX = point[0][0]
            if point[0][1] < minY:
                minY = point[0][1]

        return minX, minY

    # масштабирует контур на заданный размер
    def scaleContour(self, contour, size=100):
        contour = self.shiftContour(contour);
        maxX, maxY = self.__getMaxSide__(contour)

        # выбираем наибольшую сторону
        current_contour_size = maxX
        if maxY > current_contour_size:
            current_contour_size = maxY

        scaleFactor  = size / current_contour_size

        for point in contour:
            point[0][0] -= int(point[0][0] * scaleFactor)
            point[0][1] -= int(point[0][1] * scaleFactor)

        return contour

    # смещает контур на начало координат
    def shiftContour(self, contour):
        minX, minY = self.__getMinSide__(contour)

        for point in contour:
            point[0][0] -= minX
            point[0][1] -= minY

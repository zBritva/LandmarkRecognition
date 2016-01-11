# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np
from lib import h_mark_processor

img = cv2.imread('Marks/H-helipad.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gray_tr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#                                  cv2.THRESH_BINARY, 11, 2)
_, gray_tr = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

img2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, 0, (0, 0, 255), 2)

cp = h_mark_processor.HMarkProcessor()

h_mark_points = cp.getBoxROI([[105, 70], [105, len(img[0])-70], [len(img[0])-95, len(img[1])-70], [len(img[1])-95, 70]])

result = cp.checkBoxROIToHMark(img, h_mark_points)

cv2.circle(img, (h_mark_points[0][0], h_mark_points[0][1]), 3, (255, 0, 0), 5)
cv2.circle(img, (h_mark_points[1][0], h_mark_points[1][1]), 3, (0, 255, 0), 5)
cv2.circle(img, (h_mark_points[2][0], h_mark_points[2][1]), 3, (0, 0, 255), 5)
cv2.circle(img, (h_mark_points[3][0], h_mark_points[3][1]), 3, (255, 255, 0), 5)
cv2.circle(img, (h_mark_points[4][0], h_mark_points[4][1]), 3, (255, 0, 255), 5)
cv2.circle(img, (h_mark_points[5][0], h_mark_points[5][1]), 3, (0, 255, 255), 5)
cv2.circle(img, (h_mark_points[6][0], h_mark_points[6][1]), 3, (255, 100, 255), 5)
cv2.circle(img, (h_mark_points[7][0], h_mark_points[7][1]), 3, (100, 255, 255), 5)

print result



cv2.imshow('frame', img)

while (True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

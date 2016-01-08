# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np
from time import sleep
from lib.contour_processor import ContourProcessor

# camera
cap = cv2.VideoCapture(0)

# image data of leard (helipad, landmark)
cascade = cv2.CascadeClassifier('./train_data/train3/H-helipad-2/cascade.xml')

kernel = np.ones((3, 3), np.uint8)

if cascade.empty():
    print 'CASCADE IS EMPTY'
    exit(-1)

# print 'HAAR' if cascade.getFeatureType() == 0 else 'LBP'

contourProcessor = ContourProcessor()

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # cv2.imshow('frame', frame)

    #
    try:
        # print 'frame'
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        print 'Get frame ERROR'
        continue
    # gray = (255-gray)
    #
    # gray = frame
    #

    # gray = cv2.medianBlur(gray, 3)


    # # convert to im = np.array(float_img * 255, dtype = np.uint8)
    # gray = np.array(gray * 255, dtype=np.uint8)

    # gray = (255-gray)


    gray_tr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY, 11, 2)

    opening = cv2.morphologyEx(gray_tr, cv2.MORPH_OPEN, kernel, iterations=1)
    gray_res = cv2.dilate(opening, kernel, iterations=1)

    im2, contours, hierarchy = cv2.findContours(gray_res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    result = contourProcessor.findHMarkhelipad(frame, contours)


    mark_position = contourProcessor.findHMarkhelipad(frame, contours)
    # print 'Contours count: ', len(contours)

    # gray = cv2.medianBlur(gray, 3)



    frameOut = gray
    #
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

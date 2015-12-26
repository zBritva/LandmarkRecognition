# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np
from time import sleep

# camera
cap = cv2.VideoCapture(0)

# image data of leard (helipad, landmark)
cascade = cv2.CascadeClassifier('./train_data/train3/H-helipad/cascade.xml')

kernel = np.ones((3, 3), np.uint8)

if cascade.empty():
    print 'CASCADE IS EMPTY'
    exit(-1)

# print 'HAAR' if cascade.getFeatureType() == 0 else 'LBP'



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

    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)

    # # sure background area
    # gray = cv2.dilate(opening, kernel, iterations=1)

    # # convert to im = np.array(float_img * 255, dtype = np.uint8)
    # gray = np.array(gray * 255, dtype=np.uint8)

    # im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    # gray = (255-gray)


    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                 cv2.THRESH_BINARY, 11, 2)

    # gray = cv2.medianBlur(gray, 3)
    marks = cascade.detectMultiScale(gray, 3.8, 3)

    # filter detections

    for (x, y, w, h) in marks:
        roi = gray[y:y+h, x:x+w]

        roi = cv2.medianBlur(roi, 3)

        _, roi_gray = cv2.threshold(roi, 70, 225, cv2.THRESH_BINARY_INV)

        opening = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, kernel, iterations=1)
        roi_gray = cv2.dilate(opening, kernel, iterations=1)

    #
        # cv2.imshow('rio', roi_gray)
    #
    #     break
    #
    #     print 'rio'
    #
    #     # sleep(15)
    #
    # sleep(15)

    # end filter of detection


    for (x, y, w, h) in marks:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ret,frameOut = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

    frameOut = gray
    #
    cv2.imshow('frame', frameOut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

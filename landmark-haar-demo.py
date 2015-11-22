# -*- coding: utf-8 -*-
__author__ = 'zBritva'


import cv2
import numpy as np


# camera
cap = cv2.VideoCapture(0)


# image of land zone (H-mark, landmark)
land_zone = cv2.imread('samples/land-zone-raspberry.jpg')

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    frameOut = frame

    cv2.imshow('frame', frameOut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
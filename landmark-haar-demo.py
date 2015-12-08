# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np


# camera
cap = cv2.VideoCapture(0)


# image data of leard (helipad, landmark)
cascade = cv2.CascadeClassifier('data/cascade.xml')

print 'HAAR' if cascade.getFeatureType() == 0 else 'LBP'

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # _, gray = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    # gray = cv2.medianBlur(gray, 7)

    # convert to im = np.array(float_img * 255, dtype = np.uint8)
    # gray = np.array(gray * 255, dtype=np.uint8)

    # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                              cv2.THRESH_BINARY, 11, 2)

    # gray = cv2.medianBlur(gray, 3)
    faces = cascade.detectMultiScale(gray, 1.8, 3)

    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 0), 2)

    # ret,frameOut = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

    frameOut = gray

    cv2.imshow('frame', frameOut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

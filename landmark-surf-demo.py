# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
from cv2 import xfeatures2d

cap = cv2.VideoCapture(0)

surf = xfeatures2d.SURF_create()



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    kp, des = surf.detectAndCompute(frame, None)

    frameOut = cv2.drawKeypoints(frame, kp, None, (255, 0, 0), 4)

    # Display the resulting frame
    cv2.imshow('frame', frameOut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

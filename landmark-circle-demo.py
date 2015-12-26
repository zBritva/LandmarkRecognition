# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np

# camera
cap = cv2.VideoCapture(0)

# print 'HAAR' if cascade.getFeatureType() == 0 else 'LBP'

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    frameOut = frame
    # detect circles in the image
    circles = cv2.HoughCircles(gray, 3, 1, 15)

    if circles != None:
        width, height = frameOut.shape[:2]
        cv2.rectangle(frameOut, (0, 0), (height, width), (0, 255, 255), 50 )

        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            # draw the outer circle
            cv2.circle(frameOut, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(frameOut, (circle[0], circle[1]), 2, (0, 0, 255), 3)

    cv2.imshow('frame', frameOut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

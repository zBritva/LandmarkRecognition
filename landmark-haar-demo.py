# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np


# camera
cap = cv2.VideoCapture(0)


# image data of leard (helipad, landmark)
face_cascade = cv2.CascadeClassifier('train_data/train2/results/cascade.xml')

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    frameOut = frame

    cv2.imshow('frame', frameOut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

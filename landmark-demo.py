# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np
from math import fabs
from lib import controur_processor
from lib import h_mark_processor
from lib import e_mark_processor
from lib import z_mark_processor

# camera
cap = cv2.VideoCapture(0)

# check camera
check, test_frame = cap.read()

if not check:
    print 'Camera not found'
    exit()

print 'FRAME SIZE:' + str(len(test_frame[0])) + ' ' + str(len(test_frame[1]))

hmark = h_mark_processor.HMarkProcessor()
emark = e_mark_processor.EMarkProcessor()
zmark = z_mark_processor.ZMarkProcessor()

kernel = np.ones((3, 3), np.uint8)

cv2.namedWindow("frame1", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("frame2", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("frame3", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("frame4", cv2.WINDOW_AUTOSIZE)

display_frame_size = (320, 240)

while (True):
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            continue

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            print 'Get frame ERROR'
            continue

        # simple
        gray_tr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_hight = gray_tr
        gray_medium = np.copy(gray_tr)
        gray_low = np.copy(gray_tr)

        _, binary_double_hight = cv2.threshold(gray_hight, 175, 255, cv2.THRESH_BINARY)
        _, binary_hight = cv2.threshold(gray_hight, 155, 225, cv2.THRESH_BINARY)
        _, binary_medium = cv2.threshold(gray_medium, 125, 200, cv2.THRESH_BINARY)
        _, binary_low = cv2.threshold(gray_low, 75, 150, cv2.THRESH_BINARY)
        _, binary_double_low = cv2.threshold(gray_low, 25, 125, cv2.THRESH_BINARY)

        cv2.imshow('GR', gray_tr)
        cv2.imshow('BDH', binary_double_hight)
        cv2.imshow('BH', binary_hight)
        cv2.imshow('BM', binary_medium)
        cv2.imshow('BL', binary_low)
        cv2.imshow('BDL', binary_double_low)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        continue

        opening = cv2.morphologyEx(binary_hight, cv2.MORPH_OPEN, kernel, iterations=1)
        bin_res = cv2.dilate(opening, kernel, iterations=1)

        binary_result = cv2.medianBlur(bin_res, 3)

        # CIRCLE MODE
        im2, contours, hierarchy = cv2.findContours(binary_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour = cv2.approxPolyDP(contour, 2, True)

            # find circle
            area = cv2.contourArea(contour, oriented=False)
            perim = cv2.arcLength(contour, closed=True)
            ratio = None
            if perim > 0:
                ratio = area / (perim * perim)

            if perim > 0 and 0.07 < ratio < 0.087:
                x, y, width, height = cv2.boundingRect(contour)
                roi_frame = np.copy(frame[y:y + height, x:x + width])
                roi_circle = np.copy(binary_hight[y:y + height, x:x + width])


                tt = None
                circle_inner_contours = None
                circle_inner_contours_hierarchy = None
                try:
                    tt, circle_inner_contours, circle_inner_contours_hierarchy = cv2.findContours(
                        np.copy(roi_circle),
                        cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE)
                except Exception as ex:
                    print ex
                    continue

                center, radius = cv2.minEnclosingCircle(contour)
                cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 3)

                # display distance
                # distance_to_mark = hmark.calculateDistance(radius)
                # distance_to_mark = int(distance_to_mark)

                # cv2.putText(frame, 'DISTANCE: ' + str(distance_to_mark) + ' cm',
                #             (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 50, 255))

                cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0, 0, 255), 3)

                for contour in circle_inner_contours:
                    contour = cv2.approxPolyDP(contour, 2, True)

                    # TODO обрабатывать только контуры, точки которых находятся внутри круга
                    shifted_center = (center[0] - x, center[1] - y)
                    if len(contour) < 5 or fabs(cv2.arcLength(contour, True)) < 20:
                        continue

                    rect = cv2.minAreaRect(contour)

                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    h_mark_points = hmark.getBoxROI(box)
                    result = hmark.checkBoxROIToHMark(roi_circle, h_mark_points, True, 95)

                    if result:
                        cv2.drawContours(roi_frame, [box], 0, (0, 255, 0), 2)
                        hmark.drawMarkType(frame)
                        hmark.drawMark(roi_frame, h_mark_points, (x, y))

                    # Check to E mark
                    e_mark_points = emark.getBoxROI(box)
                    result = emark.checkBoxROIToHMark(roi_circle, e_mark_points, True)

                    if result:
                        cv2.drawContours(roi_frame, [box], 0, (0, 255, 0), 2)
                        emark.drawMarkType(frame)
                        emark.drawMark(roi_frame, h_mark_points, (x, y))

                    # Check to Z mark
                    z_mark_points = zmark.getBoxROI(box)
                    result = zmark.checkBoxROIToHMark(roi_circle, z_mark_points, True)

                    if result:
                        zmark.drawMarkType(frame)
                        cv2.drawContours(roi_frame, [box], 0, (0, 255, 0), 2)
                        zmark.drawMark(roi_frame, z_mark_points, (x, y))

                    cv2.imshow('frame3', cv2.resize(roi_frame, display_frame_size))


                cv2.imshow('frame4', cv2.resize(roi_circle, display_frame_size))

        cv2.imshow('frame2', cv2.resize(binary_hight, display_frame_size))
        cv2.imshow('frame1', cv2.resize(frame, display_frame_size))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as ex:
        print ex

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

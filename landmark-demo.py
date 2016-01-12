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

# image data of leard (helipad, landmark)
cascade = cv2.CascadeClassifier('./train_data/train3/H-helipad-2/cascade.xml')
hmark = h_mark_processor.HMarkProcessor()
emark = e_mark_processor.EMarkProcessor()
zmark = z_mark_processor.ZMarkProcessor()

kernel = np.ones((3, 3), np.uint8)

if cascade.empty():
    print 'CASCADE IS EMPTY'
    exit(-1)

# print 'HAAR' if cascade.getFeatureType() == 0 else 'LBP'

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

        # simple
        gray_tr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray_tr, 127, 255, cv2.THRESH_BINARY)

        # adaptive
        # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        #                              cv2.THRESH_BINARY, 11, 2)


        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        bin_res = cv2.dilate(opening, kernel, iterations=1)

        binary_result = cv2.medianBlur(bin_res, 3)
        marks = cascade.detectMultiScale(binary_result, 3.8, 3)

        # filter detections

        # for (x, y, w, h) in marks:
        #     roi = gray[y:y+h, x:x+w]
        #
        #     roi = cv2.medianBlur(roi, 3)
        #
        #     _, roi_gray = cv2.threshold(roi, 70, 225, cv2.THRESH_BINARY_INV)
        #
        #     opening = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, kernel, iterations=1)
        #     roi_gray = cv2.dilate(opening, kernel, iterations=1)

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

        # HAAR mode
        # if False:
        #
        #     for (x, y, w, h) in marks:
        #         # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #
        #         roi = frame[y:y + h, x:x + w]
        #
        #         # debug
        #         # while(True):
        #         #     if cv2.waitKey(1) & 0xFF == ord('c'):
        #         #         break
        #
        #         gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #
        #         _, binary_roi = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
        #
        #         cv2.imshow('frame3', roi)
        #         cv2.imshow('frame4', binary_roi)
        #
        #         im2, contours, hierarchy = cv2.findContours(binary_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #
        #         for contour in contours:
        #             contour = cv2.approxPolyDP(contour, 2, True)
        #
        #             if len(contour) > 5 and fabs(cv2.arcLength(contour, True)) > 20:
        #                 # x, y, width, height = cv2.boundingRect(contour)
        #                 # roi = frame[y:y+height, x:x+width]
        #                 # shift contour
        #                 # contour = cp.shiftContour(contour, x, y)
        #
        #                 rect = cv2.minAreaRect(contour)
        #
        #                 box = cv2.boxPoints(rect)
        #                 box = np.int0(box)
        #                 # cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        #                 # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
        #
        #
        #                 # Check to H mark
        #                 h_mark_points = hmark.getBoxROI(box)
        #                 result = hmark.checkBoxROIToHMark(binary_roi, h_mark_points, True)
        #
        #                 if result:
        #                     cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        #                 else:
        #                     cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        #
        #                     hmark.drawMark(frame, h_mark_points, (x, y))
        #
        #                 # Check to E mark
        #                 e_mark_points = emark.getBoxROI(box)
        #                 result = emark.checkBoxROIToHMark(binary_roi, e_mark_points, True)
        #
        #                 if result:
        #                     cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        #                 else:
        #                     cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        #
        #                     emark.drawMark(frame, h_mark_points, (x, y))

        # ret,frameOut = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

        # gray_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # _, binary_roi = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)

        # CIRCLE MODE
        if True:
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
                    roi_circle = np.copy(binary[y:y + height, x:x + width])


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

                    # cv2.imshow('frame4', tt)

                    center, radius = cv2.minEnclosingCircle(contour)
                    cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 3)

                    # display distance
                    distance_to_mark = hmark.calculateDistance(radius)
                    distance_to_mark = int(distance_to_mark)

                    cv2.putText(frame, 'DISTANCE: ' + str(distance_to_mark) + ' cm',
                                (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 50, 255))

                    # cv2.putText(frame, 'X:' + str(x) + 'Y:' + str(y), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    #             (0, 0, 255))
                    cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0, 0, 255), 3)

                    # cv2.drawContours(frame, circle_inner_contours, 0, (0, 255, 0), 2)

                    for contour in circle_inner_contours:
                        contour = cv2.approxPolyDP(contour, 2, True)

                        # for point in contour:
                        #     cv2.putText(frame, 'X:' + str(point[0][0]) + 'Y:' + str(point[0][1]),
                        #                 (point[0][0], point[0][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.3, (0, 0, 55))

                        # TODO обрабатывать только контуры, точки которых находятся внутри круга
                        shifted_center = (center[0] - x, center[1] - y)
                        if len(contour) < 5 or fabs(cv2.arcLength(contour, True)) < 20:
                            continue

                        # or not cp.isInCircleInside(contour,
                        #                                 center,
                        #                                 radius)

                        # if len(contour) > 5 and fabs(cv2.arcLength(contour, True)) > 20:
                        # x, y, width, height = cv2.boundingRect(contour)
                        # roi = frame[y:y+height, x:x+width]
                        # shift contour
                        # contour = cp.shiftContour(contour, x, y)

                        rect = cv2.minAreaRect(contour)

                        box = cv2.boxPoints(rect)
                        box = np.int0(box)

                        h_mark_points = hmark.getBoxROI(box)
                        result = hmark.checkBoxROIToHMark(roi_circle, h_mark_points, True, 95)

                        if result:
                            cv2.drawContours(roi_frame, [box], 0, (0, 255, 0), 2)
                            hmark.drawMarkType(frame)
                            hmark.drawMark(roi_frame, h_mark_points, (0, 0))
                        # else:
                        #     cv2.drawContours(roi_frame, [box], 0, (0, 0, 255), 2)
                        #     hmark.drawMark(roi_frame, h_mark_points, (0, 0))

                        # Check to E mark
                        e_mark_points = emark.getBoxROI(box)
                        result = emark.checkBoxROIToHMark(roi_circle, e_mark_points, True)

                        if result:
                            cv2.drawContours(roi_frame, [box], 0, (0, 255, 0), 2)
                            emark.drawMarkType(frame)
                            emark.drawMark(roi_frame, h_mark_points, (x, y))
                        # else:
                        #     cv2.drawContours(roi_frame_e, [box], 0, (0, 0, 255), 2)
                        #     emark.drawMark(roi_frame_e, h_mark_points, (x, y))

                        # Check to Z mark
                        z_mark_points = zmark.getBoxROI(box)
                        result = zmark.checkBoxROIToHMark(roi_circle, z_mark_points, True)

                        if result:
                            zmark.drawMarkType(frame)
                            cv2.drawContours(roi_frame, [box], 0, (0, 255, 0), 2)
                            zmark.drawMark(roi_frame, z_mark_points, (x, y))
                        # else:
                        #     cv2.drawContours(roi_frame_e, [box], 0, (0, 0, 255), 2)
                        #     emark.drawMark(roi_frame_e, h_mark_points, (x, y))

                        cv2.imshow('frame3', cv2.resize(roi_frame, display_frame_size))


                    cv2.imshow('frame4', cv2.resize(roi_circle, display_frame_size))


        frameOut = frame
        #
        cv2.imshow('frame2', cv2.resize(binary, display_frame_size))
        cv2.imshow('frame1', cv2.resize(frame, display_frame_size))

        # cv2.resizeWindow('frame1', 150, 150)
        # cv2.resizeWindow('frame2', 150, 150)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as ex:
        print ex

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

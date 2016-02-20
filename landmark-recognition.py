# -*- coding: utf-8 -*-
__author__ = 'zBritva'

from math import fabs
import os
import sys

import cv2
import numpy as np
import time
from marktypes import e_mark_processor
from marktypes import h_mark_processor
from marktypes import z_mark_processor
from landmark_config import LandmarkRecognitionConfiguration
from landmark_frame import LandmarkFrame


class HEZDetector:
    def __init__(self):
        self.mark_positions = list()

    def process(self):
        prev_time = time.time()

        # camera
        cap = cv2.VideoCapture(0)

        # check camera
        check, test_frame = cap.read()

        # read the configuration
        lrc = LandmarkRecognitionConfiguration()
        lf = LandmarkFrame(lrc)

        if not check:
            print 'Camera not found'
            exit(-1)

        print 'FRAME SIZE:' + str(len(test_frame[0])) + ' ' + str(len(test_frame[1]))

        hmark = h_mark_processor.HMarkProcessor()
        emark = e_mark_processor.EMarkProcessor()
        zmark = z_mark_processor.ZMarkProcessor()

        SEARCH_ALL_LANDMARKS = lrc.get_find_mode()

        kernel = np.ones((3, 3), np.uint8)

        cv2.namedWindow("frame1", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("frame2", cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow("frame2_bin0", cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow("frame2_bin1", cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow("frame2_bin2", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("frame3", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("frame4", cv2.WINDOW_AUTOSIZE)

        display_frame_size = lrc.get_frame_size()

        while (True):
            try:
                cur_time = time.time()
                if cur_time - prev_time < 0.5:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    time.sleep(0.5)
                    continue

                prev_time = cur_time

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

                # binary_images = lf.get_frames(gray_tr)

                # BEGIN PROCESSING CONTOURS FOR CURRENT FRAME
                # for binary_source_image in binary_images:
                _, binary_source_image = cv2.threshold(gray_tr, 155, 255, cv2.THRESH_BINARY)

                # flag about that mark was found or not
                mark_found = False

                # morphology extraction and dilate image for reduce noise in binary image
                # to remove dots in image for reduce contour count to processing
                opening = cv2.morphologyEx(binary_source_image, cv2.MORPH_OPEN, kernel, iterations=1)
                bin_res = cv2.dilate(opening, kernel, iterations=1)
                binary_result = cv2.medianBlur(bin_res, 3)

                # WE FIND ALL CONTOURS IN IMAGE TO PROCESS

                #3.0.0. im2, contours, hierarchy = cv2.findContours(binary_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours, _ = cv2.findContours(binary_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                # PROCESSING ALL CONTOURS FOR CURRENT FRAME
                for contour in contours:
                    # mark_found = False
                    contour = cv2.approxPolyDP(contour, 2, False)

                    # finding circle by geometrically criteria
                    area = cv2.contourArea(contour, oriented=False)
                    perim = cv2.arcLength(contour, closed=True)
                    ratio = None
                    if perim > 0:
                        ratio = area / (perim * perim)

                    # WAS FOUND CONTOUR WHICH LOOKS LIKE CIRCLE
                    if perim > 0 and 0.07 < ratio < 0.087:
                        x, y, width, height = cv2.boundingRect(contour)
                        roi_frame = np.copy(frame[y:y + height, x:x + width])
                        roi_circle = np.copy(binary_source_image[y:y + height, x:x + width])

                        tt = None
                        circle_inner_contours = None
                        circle_inner_contours_hierarchy = None
                        # just in try except,
                        # because some time we can get error durning work of cv2.findContours function
                        try:
                            #3.0.0. tt, circle_inner_contours, circle_inner_contours_hierarchy = cv2.findContours(
                            circle_inner_contours, circle_inner_contours_hierarchy = cv2.findContours(
                                    np.copy(roi_circle),
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
                        except Exception as ex:
                            print ex
                            continue

                        # draw contour for visual debugging
                        center, radius = cv2.minEnclosingCircle(contour)
                        cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (255, 0, 255), 3)

                        # display distance
                        # distance_to_mark = hmark.calculateDistance(radius)
                        # distance_to_mark = int(distance_to_mark)

                        # cv2.putText(frame, 'DISTANCE: ' + str(distance_to_mark) + ' cm',
                        #             (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 50, 255))

                        # PROCESS ALL CONTOURS IN CIRCLE
                        for contour_in_circle in circle_inner_contours:
                            contour_in_circle = cv2.approxPolyDP(contour_in_circle, 2, True)

                            # TODO обрабатывать только контуры, точки которых находятся внутри круга
                            shifted_center = (center[0] - x, center[1] - y)
                            if len(contour_in_circle) < 5 or fabs(cv2.arcLength(contour_in_circle, True)) < 20:
                                continue

                            cv2.circle(binary_source_image, (int(center[0]), int(center[1])), int(radius), (0, 0, 255),
                                       3)

                            # we get the minimum rectangle covering the circuit
                            rect = cv2.minAreaRect(contour_in_circle)
                            #3.0.0 box = cv2.cv.boxPoints(rect)
                            box = cv2.cv.BoxPoints(rect)
                            box = np.int0(box)

                            # check to H mark
                            h_mark_points = hmark.getBoxROI(box)
                            result_h_mark = hmark.checkBoxROIToHMark(roi_circle, h_mark_points, True, 95)

                            if result_h_mark:
                                cv2.drawContours(roi_frame, [box], 0, (0, 255, 0), 2)
                                hmark.drawMarkType(frame)
                                hmark.drawMark(roi_frame, h_mark_points, (x, y))

                            # check to E mark
                            e_mark_points = emark.getBoxROI(box)
                            result_e_mark = emark.checkBoxROIToHMark(roi_circle, e_mark_points, True)

                            if result_e_mark:
                                cv2.drawContours(roi_frame, [box], 0, (0, 255, 0), 2)
                                emark.drawMarkType(frame)
                                emark.drawMark(roi_frame, h_mark_points, (x, y))

                            # Check to Z mark
                            z_mark_points = zmark.getBoxROI(box)
                            result_z_mark = zmark.checkBoxROIToHMark(roi_circle, z_mark_points, True)

                            if result_z_mark:
                                zmark.drawMarkType(frame)
                                cv2.drawContours(roi_frame, [box], 0, (0, 255, 0), 2)
                                zmark.drawMark(roi_frame, z_mark_points, (x, y))

                            # cv2.imshow('frame3', cv2.resize(roi_frame, display_frame_size))

                            if result_z_mark or result_e_mark or result_h_mark:
                                mark_found = True  # set that mark found in current frame, skip all other contours
                                break

                        # draw cirle if mark was found
                        if mark_found:
                            cv2.circle(frame, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 3)
                            cv2.circle(frame, (int(center[0]), int(center[1])), 1, (0, 0, 255), 3)

                        # when all contours was processed check the result,
                        # if found and mode was set to seach only first mark skip other contours
                        if mark_found and SEARCH_ALL_LANDMARKS:
                            break  # break the main loop of contour processing, because mark was found for current frame

                            # cv2.imshow('frame4', cv2.resize(roi_circle, display_frame_size))
                # cv2.imshow('frame2', cv2.resize(binary_source_image, display_frame_size))

                # cv2.imshow('frame2_bin0', cv2.resize(binary_images[0], display_frame_size))
                # cv2.imshow('frame2_bin1', cv2.resize(binary_images[1], display_frame_size))
                # cv2.imshow('frame2_bin2', cv2.resize(binary_images[2], display_frame_size))

                # END PROCESSING CONTOURS FOR CURRENT FRAME
                cv2.imshow('frame1', cv2.resize(frame, display_frame_size))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as ex:
                print ex
        print ex.args

        # When everything done, release the capture
        cap.release()
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    process = HEZDetector()
    process.process()

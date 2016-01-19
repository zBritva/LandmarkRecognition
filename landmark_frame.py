# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np

class LandmarkFrame:
    """
    This class process source grayscale image and give thre binary images for processing
    corresponding mode paramenter
    """

    def __init__(self, config):
        self.config = config

        self.binary_upper = self.config.get_canny_binary_upper()
        self.binary_up = self.config.get_canny_binary_up()
        self.binary_medium = self.config.get_canny_binary_medium()
        self.binary_low = self.config.get_canny_binary_low()
        self.binary_lower = self.config.get_canny_binary_lower()
        self.mode = self.config.get_mode()

    def get_frames(self, img):
        # gray_hight = img
        gray_hight = np.copy(img)
        gray_medium = np.copy(img)
        gray_low = np.copy(img)

        if self.mode == 'DAY':
            _, binary_upper = cv2.threshold(gray_hight, self.binary_upper[0], self.binary_upper[1], cv2.THRESH_BINARY)
            _, binary_up = cv2.threshold(gray_medium, self.binary_up[0], self.binary_up[1], cv2.THRESH_BINARY)
            _, binary_medium = cv2.threshold(gray_low, self.binary_medium[0], self.binary_medium[1], cv2.THRESH_BINARY)

            return [binary_upper, binary_up, binary_medium]

        if self.mode == 'NIGHTFALL':
            _, binary_up = cv2.threshold(gray_hight, self.binary_up[0], self.binary_up[1], cv2.THRESH_BINARY)
            _, binary_medium = cv2.threshold(gray_medium, self.binary_medium[0], self.binary_medium[1],
                                             cv2.THRESH_BINARY)
            _, binary_low = cv2.threshold(gray_low, self.binary_low[0], self.binary_low[1], cv2.THRESH_BINARY)

            return [binary_up, binary_medium, binary_low]

        if self.mode == 'NIGHT':
            _, binary_medium = cv2.threshold(gray_hight, self.binary_medium[0], self.binary_medium[1],
                                             cv2.THRESH_BINARY)
            _, binary_low = cv2.threshold(gray_medium, self.binary_low[0], self.binary_low[1], cv2.THRESH_BINARY)
            _, binary_lower = cv2.threshold(gray_low, self.binary_lower[0], self.binary_lower[1], cv2.THRESH_BINARY)

            return [binary_medium, binary_low, binary_lower]

        raise Exception('Incorrect config "mode" option')

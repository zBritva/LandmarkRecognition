# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import ConfigParser
import os


class LandmarkRecognitionConfiguration:
    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.configPath = os.path.dirname(__file__) + 'config/config.ini'
        self.config.read(self.configPath)

    def get_mode(self):
        """
        Return mode for image processing. This parameter using
        for select optimal values of canny operator for binarization image
        :rtype: str
        """
        try:
            return self.config.get('MAIN', 'MODE')
        except Exception:
            return 'DAY'

    def get_frame_size(self):
        """
        Return frame size to display image example in test mode
        :return: tuple
        """
        try:
            width = self.config.get('TEST', 'DISPLAY_WIDTH')
            height = self.config.get('TEST', 'DISPLAY_HEIGHT')
            return width, height
        except Exception:
            return 320, 240

    def get_work_mode(self):
        """
        This configuration control work mode of script:
        ACTIVE - script give current coordinates and mark type to another process
        PASSIVE - script give current coordinates and mark type to another process by request
        :return:
        """
        try:
            return self.config.get('MAIN', 'WORK_MODE')
        except Exception:
            return 'PASSIVE'

    def get_log_mode(self):
        """
        Logging of any messages of script: warnings and errors
        :return:
        """
        try:
            return self.config.get('TEST', 'LOG')
        except Exception:
            return 'OFF'

    def get_canny_binary_upper(self):
        """

        :return:
        """
        try:
            str = self.config.get('CORE', 'binary_upper')
            ls = str.split(',')
            return tuple(ls)
        except Exception:
            return 175, 255

    def get_canny_binary_up(self):
        try:
            str = self.config.get('CORE', 'binary_up')
            ls = str.split(',')
            return tuple(ls)
        except Exception:
            return 155, 225

    def get_canny_binary_medium(self):
        try:
            str = self.config.get('CORE', 'binary_medium')
            ls = str.split(',')
            return tuple(ls)
        except Exception:
            return 125, 200

    def get_canny_binary_low(self):
        try:
            str = self.config.get('CORE', 'binary_low')
            ls = str.split(',')
            return tuple(ls)
        except Exception:
            return 75, 150

    def get_canny_binary_lower(self):
        try:
            str = self.config.get('CORE', 'binary_lower')
            ls = str.split(',')
            return tuple(ls)
        except Exception:
            return 25, 125

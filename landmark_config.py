# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import ConfigParser
import os


class LandmarkRecognitionConfiguration:
    """
    Class provide loading of config file (config/config.ini)
    """

    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.configPath = os.path.dirname(__file__) + '/config/config.ini'
        self.config.read(self.configPath)

    def get_mode(self):
        """
        Return mode for image processing. This parameter using
        for select optimal values of canny operator for binarization image

        :return: Config value
        """
        try:
            return self.config.get('MAIN', 'MODE').strip()
        except Exception:
            return 'DAY'

    def get_frame_size(self):
        """
        Return frame size to display image example in test mode

        :return: Config value
        """
        try:
            width = self.config.get('TEST', 'DISPLAY_WIDTH')
            height = self.config.get('TEST', 'DISPLAY_HEIGHT')
            return int(width), int(height)
        except Exception:
            return 320, 240

    def get_work_mode(self):
        """
        This configuration control work mode of script:
        ACTIVE - script give current coordinates and mark type to another process
        PASSIVE - script give current coordinates and mark type to another process by request

        :return: Config value
        """
        try:
            return self.config.get('MAIN', 'WORK_MODE')
        except Exception:
            return 'PASSIVE'

    def get_log_mode(self):
        """
        Logging of any messages of script: warnings and errors

        :return: Config value
        """
        try:
            return True if self.config.get('TEST', 'LOG').upper().strip() == 'ON' else False
        except Exception:
            return 'OFF'

    def get_canny_binary_upper(self):
        """
        Load canny limits for upper range

        :return: Config value
        """
        try:
            str = self.config.get('CORE', 'binary_upper')
            ls = str.split(',')
            return int(ls[0]), int(ls[1])
        except Exception:
            return 175, 255

    def get_canny_binary_up(self):
        """
        Load canny limits for up range

        :return: Config value
        """
        try:
            str = self.config.get('CORE', 'binary_up')
            ls = str.split(',')
            return int(ls[0]), int(ls[1])
        except Exception:
            return 155, 255

    def get_canny_binary_medium(self):
        """
        Load canny limits for medium range

        :return: Config value
        """
        try:
            str = self.config.get('CORE', 'binary_medium')
            ls = str.split(',')
            return int(ls[0]), int(ls[1])
        except Exception:
            return 125, 255

    def get_canny_binary_low(self):
        """
        Load canny limits for low range

        :return: Config value
        """
        try:
            str = self.config.get('CORE', 'binary_low')
            ls = str.split(',')
            return int(ls[0]), int(ls[1])
        except Exception:
            return 75, 255

    def get_canny_binary_lower(self):
        """
        Load canny limits for lower range

        :return: Config value
        """
        try:
            str = self.config.get('CORE', 'binary_lower')
            ls = str.split(',')
            return int(ls[0]), int(ls[1])
        except Exception:
            return 25, 255

    def get_find_mode(self):
        """
        This parameter set will script search all landmarks for current frame by processing
        all founded contours. Or skip all if first was found

        :return: Config value

        :rtype: Boolean
        """
        try:
            str = self.config.get('MAIN', 'SEARCH_ALL_LANDMARKS')
            str = str.upper()
            if str == 'TRUE':
                return True
            else:
                return False
        except Exception:
            return False

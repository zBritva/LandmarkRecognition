.. _Classes-label:

Classes
~~~~~~~

.. _ContourProcessor-label:

..

.. py:class:: ContourProcessor

This class contains methods for processing frame image for recognizing landmark

.. py:method:: __distance__(self, point1, point2)

   Function calculates distance between two points.

.. py:method:: __middlePoint__(self, point1, point2, ratio=1)

   Function returns coordinates of point in line which pass throught point1 and point2. And point divides the line in a predetermined ratio

.. py:method:: createLineIterator(self, point1, point2, img, binary=False)

   Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

   Source code was given there: http://stackoverflow.com/a/32857432/5623063

.. py:method:: rgbToGray(self, color)

   Convert RGB color to grayscale

.. py:method:: getLineAvgColor(self, iterator, binary=False)

   Compute average value of color in array of pixels given by createLineIterator

.. py:method:: checkAvg(self, avg, accept_percentage)

   Returns true if the average value of pixels more than the accept_percentage

.. py:method:: shiftContour(self, contour, x, y)

   Shifts all the points of the circuit to the specified values x and y

.. py:method:: isInCircleInside(self, contour, center, radius)

   Returns true if the all point of contour inside of the circle

.. py:method:: calculateDistance(self, radius)

   Calculate distance from the camera to landmark bounding by circle

.. py:method:: drawMark(self, img, points, shift)

   Draw landmark linies in image

.. warning::

   Function must be overridden in child classes


.. py:class:: HMarkProcessor(ContourProcessor)

    Contains methods for determining H-mark helipad

.. py:method:: getBoxROI(self, box)

.. py:method:: checkBoxROIToHMark(self, image, h_mark_points, binary=False, accept_percentage=85)


.. py:class:: ZMarkProcessor(ContourProcessor)

    Contains methods for determining Z-mark helipad

.. py:method:: getBoxROI(self, box)

.. py:method:: checkBoxROIToHMark(self, image, h_mark_points, binary=False, accept_percentage=85)



.. py:class:: EMarkProcessor(ContourProcessor)

    Contains methods for determining E-mark helipad

.. py:method:: getBoxROI(self, box)

.. py:method:: checkBoxROIToHMark(self, image, h_mark_points, binary=False, accept_percentage=85)



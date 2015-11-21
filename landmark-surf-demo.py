# -*- coding: utf-8 -*-
__author__ = 'zBritva'

import cv2
import numpy as np
from cv2 import xfeatures2d


# camera
cap = cv2.VideoCapture(0)

# main method
surf = xfeatures2d.SURF_create()
#surf.setUpright(True)
# surf.extended = True #for expand descriptor to 128bits, default is 64

# image of land zone (H-mark, landmark)
land_zone = cv2.imread('samples/land-zone-sample-binary.jpg', 0)

# keypoint and descriptors of landmark
landmark_keypoints, landmark_descriptors = surf.detectAndCompute(land_zone, None)
# convert to float32
landmark_descriptors = np.float32(landmark_descriptors)


# Configure FLANN
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_LSH,
#                     table_number=6,  # 12
#                     key_size=12,  # 20
#                     multi_probe_level=1)  # 2
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
# FLANN
flann = cv2.FlannBasedMatcher(index_params, search_params)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # compute frame keypoints and descriptors
    frame_keypoints, frame_descriptors = surf.detectAndCompute(frame, None)
    # convert to float32
    frame_descriptors = np.float32(frame_descriptors)

    matches = flann.knnMatch(frame_descriptors, landmark_descriptors, k=2)

    frameOut = cv2.drawKeypoints(frame, frame_keypoints, None, (255, 0, 0), 4)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in xrange(len(matches))]

    # ratio test as per Lowe's paper

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    # Display the resulting frame
    frameOut = cv2.drawMatchesKnn(land_zone, landmark_keypoints, frame, frame_keypoints, matches, None, **draw_params)

    cv2.imshow('frame', frameOut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

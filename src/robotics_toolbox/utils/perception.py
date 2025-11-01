#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2025-09-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from typing import List
from numpy.typing import ArrayLike
import numpy as np
import cv2  # noqa


def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    """
    Find homography based on images containing the hoop and the hoop positions loaded from
    the hoop_positions.json file in the following format:

    [{
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093259019899434, -0.17564068853313258, 0.04918733225140541]
    },
    {
        "RPY": [-0.0005572332585040621, -3.141058227474627, 0.0005185830258253442],
        "translation_vector": [0.5093569397977782, -0.08814069881074972, 0.04918733225140541]
    },
    ...
    ]
    """
    # Check inputs
    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)

    img_centers = np.empty((len(hoop_positions), 2))

    # todo HW03: Detect circle in each image
    for i, img in enumerate(images):
        # highlight blue hoop
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert to HSV
        imgHSV = cv2.medianBlur(imgHSV, 5) # apply median blur to reduce noise
        
        # desired color: HSV = 209, 57, 45 (in standard range 0-360, 0-100, 0-100)
        # in OpenCV range: HSV = 104, 145, 114
        lower = np.array([90, 50, 90])
        upper = np.array([120, 200, 160])
        mask = cv2.inRange(imgHSV, lower, upper)
      
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=300, param1=100, param2=30, minRadius=10, maxRadius=400)               
        assert circles is not None, "No circles found"     

        circles = np.uint16(np.around(circles)) 

        circle = circles[0][0] 
        img_centers[i] = [circle[0], circle[1]]  # (x, y)

    ref_centers = np.empty((len(hoop_positions), 2))
    for i, hoop in enumerate(hoop_positions):
        pos = hoop["translation_vector"]
        ref_centers[i] = [pos[0], pos[1]] 

    homography = cv2.findHomography(img_centers, ref_centers, cv2.RANSAC)[0]
    return homography

import os
import numpy as np
import cv2

k = np.load("calibration/k.npy")
dist = np.load("calibration/dist.npy")

t_camera2robot = np.load("calibration/camera2robot.npy")
t_robot2camera = np.load("calibration/robot2camera.npy")

marker_img = cv2.imread("markers_right.png")

gray = cv2.cvtColor(marker_img, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, rejected = detector.detectMarkers(gray)


rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.04, k, dist)

print(tvec)

t1 = np.hstack([tvec[0][0], 1])
t2 = np.hstack([tvec[1][0], 1])


print(t_camera2robot @ t1)
print(t_camera2robot @ t2)
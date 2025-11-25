import numpy as np
from camera import CameraHelper
from ctu_bosch_sr450 import RobotBosch
import cv2
from utils import to_homogenous


k = np.load("calibration/k.npy")
dist = np.load("calibration/dist.npy")
t_camera2robot = np.load("calibration/camera2robot.npy")

print(t_camera2robot)

robot = RobotBosch()
robot.initialize()
cam = CameraHelper(robot, "lab")

q = robot.get_q()

new_q = robot.q_max

new_q[2] = q[2]
new_q[3] = q[3]

robot.move_to_q(new_q)
robot.wait_for_motion_stop()

img = cam.get_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, rejected = detector.detectMarkers(gray)

rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.04, k, dist)

tvec = tvec.flatten()
# tvec[2] = 1.5


print(tvec)

t_camera2marker = to_homogenous(rvec, tvec)

print(t_camera2marker @ t_camera2robot)





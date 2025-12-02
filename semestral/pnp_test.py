import numpy as np
from camera import CameraHelper
from ctu_bosch_sr450 import RobotBosch
import cv2
from utils import to_homogenous, select_q_min_rotation


k = np.load("calibration/k.npy")
dist = np.load("calibration/dist.npy")
t_camera2robot = np.load("calibration/camera2robot.npy")

robot = RobotBosch()
robot.initialize(True)
cam = CameraHelper(robot, "lab")

print("Q range: ", robot.q_min, robot.q_max, robot.q_home)

q_max = robot.q_max.copy()

q_max[3] = 0

robot.move_to_q(q_max)

print("Moving to max q: ", robot.q_max)

robot.wait_for_motion_stop()

img = cam.get_image()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, rejected = detector.detectMarkers(gray)

rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.04, k, dist)

tvec = np.asarray(tvec)

center = (tvec[0][0] + tvec[1][0]) / 2

center = np.asarray(np.hstack([center, 1]))

print("Center camera: ", center)

robot_center = list(t_camera2robot @ center)

print("Robot center: ", robot_center)

print("Adujsted robot center: ", robot_center[0], robot_center[1] - 0.135, robot_center[2] + 0.23, ((3/2) * np.pi) - np.pi * 0.3)

q = robot.ik_xyz(robot_center[0], robot_center[1] - 0.135, robot_center[2] + 0.23, ((3/2)) * np.pi - np.pi * 0.05)

print("Q: ", q)

if len(q) > 0:
    selected = list(select_q_min_rotation(q, robot.get_q()))

    print("Selected q: ", selected)

    robot.move_to_q(selected)

    robot.wait_for_motion_stop()




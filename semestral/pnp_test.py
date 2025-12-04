import numpy as np
from camera import CameraHelper
from ctu_bosch_sr450 import RobotBosch
import cv2
from utils import to_homogenous, select_q_min_rotation, HOOP_LEN, MAZE_OFFSET, ROBOT_Q_ROTATION_OFFSET, MAZE_HEIGHT
from planner import TrajectoryPlanner, MazeType

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

if len(tvec) < 2:
    print("Could not find markers")
    quit()

center = (tvec[0][0] + tvec[1][0]) / 2

center = np.asarray(np.hstack([center, 1]))

print("Center camera: ", center)

board_center = list(t_camera2robot @ center)

print("Board pose: ", board_center)

planner = TrajectoryPlanner(robot, MazeType.A, board_center)

configuration_sequnce = planner.get_q_sequence_maze_a()

for configuration in configuration_sequnce:
    print("Moving to q: ", configuration)
    robot.move_to_q(configuration)
    robot.wait_for_motion_stop()

input("press any button to reset")

for configuration in configuration_sequnce[::-1]:
    print("Moving to q: ", configuration)
    robot.move_to_q(configuration)
    robot.wait_for_motion_stop()

current = robot.get_q()

q_res = [0.5, 0.5, 0, current[3]]
robot.move_to_q(q_res)
robot.wait_for_motion_stop()

q_res[3] = 0
robot.move_to_q(q_res)
robot.wait_for_motion_stop()
robot.soft_home()
robot.wait_for_motion_stop()

import numpy as np
from camera import CameraHelper
from ctu_bosch_sr450 import RobotBosch
import cv2
import utils
from mazes import MazeType
from planner import TrajectoryPlanner
from maze_pose_solver import MazePoseSolver
from time import sleep

MAZE_TYPE = MazeType.A

# Load callibraion
k = np.load("calibration/k.npy")
dist = np.load("calibration/dist.npy")
T_camera2robot = np.load("calibration/camera2robot.npy")

# Robot init
robot = RobotBosch()
robot.initialize(True)
cam = CameraHelper(robot, "lab")

# Hide robot
q_max = robot.q_max.copy()
robot.move_to_q(q_max)
robot.wait_for_motion_stop()
sleep(1)

# Capture img
img = cam.get_image()

# Find maze pose
solver = MazePoseSolver(img, T_camera2robot, k, dist)
board_center = solver.find_base_center()
board_rotation = solver.find_base_rotation()

# Plan trajectory
planner = TrajectoryPlanner(robot, MAZE_TYPE, board_center, board_rotation)
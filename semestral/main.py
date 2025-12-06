import numpy as np
from camera import CameraHelper
from ctu_bosch_sr450 import RobotBosch
import cv2
import utils
from mazes import MazeType
from planner import TrajectoryPlanner
from maze_pose_solver import MazePoseSolver
from time import sleep

MAZE_TYPE = MazeType.C

# Load callibraion
k = np.load("calibration/k.npy")
dist = np.load("calibration/dist.npy")
T_camera2robot = np.load("calibration/camera2robot.npy")

# Robot init
robot = RobotBosch()
robot.initialize(True)
cam = CameraHelper(robot, "lab")

# Hide robot
q_hidden = robot.q_max.copy()
q_hidden[3] = 0
robot.move_to_q(q_hidden)
robot.wait_for_motion_stop()
sleep(1)

# Capture img
img = cam.get_image()

# Find maze pose
solver = MazePoseSolver(img, T_camera2robot, k, dist)
board_center = solver.find_base_center()[0:3]
board_rotation = solver.find_base_rotation()

print(board_center)

# Plan trajectory
planner = TrajectoryPlanner(robot, MAZE_TYPE, board_center, board_rotation)

configs_to_start = planner.get_q_sequence_tostart()
for config in configs_to_start:
    print("Moving to q: ", config)
    robot.move_to_q(config)
    robot.wait_for_motion_stop()

input("press any key")

configs_through_maze = planner.get_q_sequence_throughmaze()
for config in configs_through_maze:
    print("Moving to q: ", config)
    robot.move_to_q(config)
    robot.wait_for_motion_stop()

input("press any key")

# Return to start pose
for config in configs_through_maze[::-1]:
    print("Moving to q: ", config)
    robot.move_to_q(config)
    robot.wait_for_motion_stop()

robot.move_to_q(configs_to_start[-1])
robot.wait_for_motion_stop()

# End sequence
current = robot.get_q()

q_res = [0.5, 0.5, 0, current[3]]
robot.move_to_q(q_res)
robot.wait_for_motion_stop()

q_res[3] = 0
robot.move_to_q(q_res)
robot.wait_for_motion_stop()
robot.soft_home()
robot.wait_for_motion_stop()
robot.close()
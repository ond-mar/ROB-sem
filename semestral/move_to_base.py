from ctu_bosch_sr450 import RobotBosch
from find_base_center import find_base_center
import numpy as np
from utils import load_matrix
import time

robot = RobotBosch(tty_dev=None)
robot.initialize()

# Hide robot
q_hide = robot.q_max
q_hide[3] = 0

robot.move_to_q(q_hide)
robot.wait_for_motion_stop()
time.sleep(1)

# Capture image
image = robot.grab_image()

# Find base center
base_center = find_base_center(image)

# Move robot to base center
homography = load_matrix("homography.npy")
base_robot = base_center @ homography.T
base_robot = base_robot / base_robot[2]  # Normalize homogeneous coordinates

new_q = robot.ik_xyz(base_robot[0], base_robot[1], 0.35)
if len(new_q) > 0:
    robot.move_to_q(new_q[0])
    robot.wait_for_motion_stop()

robot.close()

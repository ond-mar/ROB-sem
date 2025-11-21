from ctu_bosch_sr450 import RobotBosch
from camera import CameraHelper
import numpy as np
import time
from utils import MIN_Z_Q
from datetime import datetime

robot = RobotBosch()
robot.initialize()
camera_helper = CameraHelper(robot, "hoop")

def save_matrix(name, matrix):
    np.save(name, matrix)

def load_matrix(name):
    with open(name, "rb") as f:
        return np.load(f)

def save_pose(q):
    timestamp = datetime.now().isoformat()
    camera_helper.capture_image(timestamp)
    save_matrix(timestamp, np.asarray(q))

home_q_0 = robot.get_q()[0]
home_q_3 = robot.get_q()[3]

step = 0.05
step_count = 20

q = robot.get_q()
q[2] = MIN_Z_Q

q[3] = np.pi / 2
q[0] = step * 12 + q[0]
for i in range(12, step_count):
    q[0] = q[0] + step
    q[1] = - i * step
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    time.sleep(0.5)
    save_pose(robot.get_q())


q[3] = 2 * np.pi - np.pi / 2
robot.move_to_q(q)
robot.wait_for_motion_stop()

q[0] = home_q_0
robot.move_to_q(q)
robot.wait_for_motion_stop()

q = robot.get_q()

for i in range(step_count):
    q[0] = q[0] - step
    q[1] = i * step
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    time.sleep(0.5)
    save_pose(robot.get_q())


q = robot.get_q()
q[3] = home_q_3

robot.move_to_q(q)
robot.wait_for_motion_stop()
robot.soft_home()
robot.wait_for_motion_stop()
robot.close()

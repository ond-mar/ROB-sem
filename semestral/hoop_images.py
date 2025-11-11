from ctu_bosch_sr450 import RobotBosch
from camera import CameraHelper
import numpy as np
import time

robot = RobotBosch()
robot.initialize()

home_q_0 = robot.get_q()[0]
home_q_3 = robot.get_q()[3]

camera_helper = CameraHelper(robot, "hoop")

step = 0.05
step_count = 20

q = robot.get_q()
q[3] = np.pi / 2
q[0] = step * 12 + q[0]
for i in range(12, step_count):
    q[0] = q[0] + step
    q[1] = - i * step
    q[2] = -0.23
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    time.sleep(0.5)
    camera_helper.capture_image()


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
    q[2] = -0.23
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    time.sleep(0.5)
    camera_helper.capture_image()


q = robot.get_q()
q[3] = home_q_3

robot.move_to_q(q)
robot.wait_for_motion_stop()
robot.soft_home()
robot.wait_for_motion_stop()
robot.close()

from camera import CameraHelper
from ctu_bosch_sr450 import RobotBosch

robot = RobotBosch()
robot.initialize()
cam = CameraHelper(robot, "lab")

q = robot.get_q()

new_q = robot.q_max

new_q[2] = q[2]
new_q[3] = q[3]

robot.move_to_q(new_q)
robot.wait_for_motion_stop()

cam.capture_image("tyc")

robot.close()




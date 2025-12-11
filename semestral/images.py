from ctu_bosch_sr450 import RobotBosch
from camera import CameraHelper

robot = RobotBosch()
robot.initialize(home=False)

camera_helper = CameraHelper(robot, "img")

camera_helper.capture_image_date("img")

robot.release()
robot.close()
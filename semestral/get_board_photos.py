from ctu_bosch_sr450 import RobotBosch
from camera import CameraHelper

# HIDE ROBOT #

# robot = RobotBosch()
# robot.initialize()

# q_hide = robot.q_max
# q_hide[3] = 0

# robot.move_to_q(robot.q_max)
# robot.wait_for_motion_stop()

# TAKE PHOTO #

robot = RobotBosch()
robot.initialize(home=False)

camera_helper = CameraHelper(robot, "base")
camera_helper.capture_image("base_random")
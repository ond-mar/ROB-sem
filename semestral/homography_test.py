from ctu_bosch_sr450 import RobotBosch
import numpy as np
from utils import load_matrix, MIN_Z

robot = RobotBosch(tty_dev=None)
#robot.initialize()

#print(robot.fk(robot.get_q()))

ref_x, ref_y = 0 , 0

homography = load_matrix("homography.npy")
print(homography)

vector = np.asarray([ref_x, ref_y, 1]) @ homography.T
vector = vector / vector[2] # IMPORTANT: normalize homogeneous coordinates

print(vector)

new_q = robot.ik_xyz(vector[0], vector[1], 0.3)

print(new_q)

if len(new_q) > 0:
    robot.move_to_q(new_q[0])
    robot.wait_for_motion_stop()

robot.close()

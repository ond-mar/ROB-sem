from ctu_bosch_sr450 import RobotBosch
import numpy as np
from utils import load_matrix, MIN_Z

robot = RobotBosch()
robot.initialize()

print(robot.fk(robot.get_q()))

ref_x, ref_y = 200 , 80

homography = load_matrix("homography.npy")

vector = np.asarray([ref_x, ref_y, 1]) @ homography.T

print(vector)

new_q = robot.ik_xyz(vector[0], vector[1], 0.3)

print(new_q)

if len(new_q) > 0:
    robot.move_to_q(new_q[0])
    robot.wait_for_motion_stop()

robot.close()

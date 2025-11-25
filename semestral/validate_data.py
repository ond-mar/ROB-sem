import numpy as np
from utils import load_matrix
from ctu_bosch_sr450 import RobotBosch


coord = load_matrix("hoop_images/2025-11-21T17:55:10.900173.npy")
print("Loaded coords:")
print(coord)

robot = RobotBosch(tty_dev=None)
print("FK result:")
print(robot.fk(coord))

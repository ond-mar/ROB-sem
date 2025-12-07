import numpy as np
import cv2

Q_TOLERANCE = 0.001

HOOP_LEN = 0.135

MAZE_HEIGHT = 0.20

MAZE_OFFSET = MAZE_HEIGHT + 0.025

MAX_Z = 0.5
MAX_Z_Q = 0

MIN_Z = 0.8
MIN_Z_Q = - 0.31

ROBOT_Q_ROTATION_OFFSET = -0.22

LEFT_Q = [-4.83117755e-01, -6.83617751e-01,  2.25225225e-05, -7.05254062e-04]
RIGHT_Q = [1.08819848e+00, -6.83617751e-01,  2.47747748e-05, -6.17097304e-04]

LEFT_Q_EXTREME =  [1, -1,  2.47747748e-05, 0]
LEFT_Q_MID = [0.9, -1.1, 0, 0]

RIGHT_Q_EXTREME =  [-1.33, 1.2, 0, 0]
RIGHT_Q_MID = [-1, 1.1, 0, 0]

def load_matrix(name):
    with open(name, "rb") as f:
        return np.load(f)

def to_homogenous(r, t):
    transform = np.eye(4)

    transform[:3, :3] = cv2.Rodrigues(r)[0]
    transform[:3, 3] = t.flatten()

    return transform

def select_qs_height_change_only(qs: list, cq) -> list[list]:
    possible_qs = filter(lambda q: np.isclose(cq[0], q[0]) and np.isclose(cq[1], q[0]), qs)

    return list(possible_qs)

def select_q_min_rotation(qs: list, cq: list):
    assert len(qs) > 0, "No Qs"

    best_dist = abs(cq[3] - qs[0][3])
    best_q = qs[0]

    for q in qs:
        if best_dist > abs(cq[3] - q[3]):
            best_dist = abs(cq[3] - q[3])
            best_q = q
    
    return best_q

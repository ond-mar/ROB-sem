import numpy as np
import cv2

MAX_Z = 0.5
MAX_Z_Q = 0

MIN_Z = 0.8
MIN_Z_Q = - 0.33

LEFT_Q = [-4.83117755e-01, -6.83617751e-01,  2.25225225e-05, -7.05254062e-04]
RIGHT_Q = [1.08819848e+00, -6.83617751e-01,  2.47747748e-05, -6.17097304e-04]

def load_matrix(name):
    with open(name, "rb") as f:
        return np.load(f)

def to_homogenous(r, t):
    transform = np.eye(4)

    transform[:3, :3] = cv2.Rodrigues(r)[0]
    transform[:3, 3] = t.flatten()

    return transform

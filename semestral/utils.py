import numpy as np
import cv2

MIN_Z = 0.200
MIN_Z_Q = - 0.30

def load_matrix(name):
    with open(name, "rb") as f:
        return np.load(f)

def to_homogenous(r, t):
    transform = np.eye(4)

    transform[:3, :3] = cv2.Rodrigues(r)[0]
    transform[:3, 3] = t.flatten()

    return transform

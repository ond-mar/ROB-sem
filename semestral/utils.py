import numpy as np

MIN_Z = 0.200
MIN_Z_Q = - 0.30

BASE_Z = 0.24
BASE_Z_Q = - 0.26

def load_matrix(name):
    with open(name, "rb") as f:
        return np.load(f)
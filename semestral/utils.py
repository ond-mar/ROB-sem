import numpy as np

MIN_Z = 0.200
MIN_Z_Q = - 0.30

def load_matrix(name):
    with open(name, "rb") as f:
        return np.load(f)
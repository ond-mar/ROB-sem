import numpy as np
from enum import Enum

class MazeType(Enum):
    A = 1
    B = 2
    C = 3

# Coordinates in meters starting at base
MAZE_SHAPES = {
    MazeType.A: np.array([[0, 0, 0], [0, 0, 0.2]]),
    MazeType.B: np.array([[0, 0, 0], [0, 0, 0.08], [0, 0.07, 0.15], [0, 0.07, 0.2]]),
    MazeType.C: np.array([[0, 0, 0], [0, 0, 0.05], [0, -0.05, 0.1], [0.05, -0.05, 0.15], [0.05, -0.05, 0.2] ]),
}

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mazes import MAZE_SHAPES, MazeType
from planner import TrajectoryPlanner

board_center = [0.25, 0.2, 0.2]
board_rotation = -np.pi / 4  # 45 degrees
maze_type = MazeType.C

planner = TrajectoryPlanner(None, maze_type, board_center, board_rotation)

maze_points_origin = MAZE_SHAPES[MazeType.C]
maze_points_robot = planner.maze_to_robot_points()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(maze_points_origin[:,0], maze_points_origin[:,1], maze_points_origin[:,2], marker='o')
ax.plot(maze_points_robot[:,0], maze_points_robot[:,1], maze_points_robot[:,2], marker='o')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_xlim([0, 0.5])
ax.set_ylim([-0.25, 0.25])
ax.set_zlim([0, 0.5])

plt.show()


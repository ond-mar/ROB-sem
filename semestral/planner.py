from ctu_bosch_sr450 import RobotBosch
from enum import Enum
from utils import to_homogenous, select_q_min_rotation, HOOP_LEN, MAZE_OFFSET, ROBOT_Q_ROTATION_OFFSET, MAZE_HEIGHT, Q_TOLERANCE
import numpy as np

class MazeType(Enum):
    A = 1
    B = 2
    C = 3

class TrajectoryPlanner:
    def __init__(self, robot: RobotBosch, maze_type: MazeType, board_center: list):
        self.robot = robot
        self.maze_type = maze_type
        self.board_center = board_center

    @property
    def current(self):
        return self.robot.get_q()

    def calculate_ee_start_pose(self):
        pose = self.board_center.copy()
        
        if pose[1] <= 0:
            offset = HOOP_LEN
            rotation = ((1/2)) * np.pi + ROBOT_Q_ROTATION_OFFSET
        else:
            rotation = ((3/2)) * np.pi + ROBOT_Q_ROTATION_OFFSET
            offset = -HOOP_LEN

        pose[1] += offset
        pose[2] += MAZE_OFFSET
        pose[3] = rotation

        return pose

    def __select_qs_height_change_only(self, current: list, qs: list) -> list[list]:
        print("Selecting qs with only vertical change current: ", current)
        print("Selecting qs with only vertical change from: ", qs)

        possible_qs = filter(lambda q: np.isclose(current[0], q[0], Q_TOLERANCE) and np.isclose(current[1], q[1], Q_TOLERANCE), qs)

        return list(possible_qs)

    def __select_q_min_rotation(self, current: list, qs: list):
        assert len(qs) > 0, "No Qs"
        current = self.robot.get_q()
        best_dist = abs(current[3] - qs[0][3])
        best_q = qs[0]

        for q in qs:
            if best_dist > abs(current[3] - q[3]):
                best_dist = abs(current[3] - q[3])
                best_q = q
        
        return best_q

    def calculate_ik(self, pose):
        print("Solving ik for: ", pose)

        qs = self.robot.ik_xyz(*pose).copy()

        if len(qs) == 0:
            print("Could not find IK for: ", pose)

        return qs

    def get_q_sequence_maze_a(self):
        print("Calculating trajectory for maze A")

        start_pose = self.calculate_ee_start_pose()
        
        print("Start pose: ", start_pose)
        
        start_q = self.__select_q_min_rotation(self.current, self.calculate_ik(start_pose))

        end_pose = start_pose.copy()
        end_pose[2] -= MAZE_HEIGHT

        print("End pose: ", end_pose)

        qs_height_only = self.__select_qs_height_change_only(start_q, self.calculate_ik(end_pose))

        end_q = self.__select_q_min_rotation(start_q, qs_height_only)

        configurations = [start_q, end_q]

        print("Configurations: ")
        print(configurations[0])
        print(configurations[1])

        return configurations

from ctu_bosch_sr450 import RobotBosch
from utils import to_homogenous, select_q_min_rotation, HOOP_LEN, MAZE_OFFSET, ROBOT_Q_ROTATION_OFFSET, MAZE_HEIGHT, Q_TOLERANCE, LEFT_Q_MID
import numpy as np
from mazes import MazeType, MAZE_SHAPES

from core.se3 import SE3
from core.so3 import SO3

class TrajectoryPlanner:
    def __init__(self, robot: RobotBosch, maze_type: MazeType, board_center: list, board_rotation: float = 0):
        self.robot = robot
        self.maze_type = maze_type
        self.board_center = board_center
        self.board_rotation = board_rotation

        self.offset = 0
        self.hoop_rotation = 0

        self.maze_points = self.maze_to_robot_points()

        self.last_pose = []
        self.last_q = []

    @property
    def current(self):
        return self.robot.get_q()

    def calculate_ee_start_pose(self):
        # pose = self.board_center.copy()
        pose = self.maze_points[-1].tolist()
        pose.append(0)

        if pose[1] <= 0:
            offset = HOOP_LEN
            rotation = ((1/2)) * np.pi + ROBOT_Q_ROTATION_OFFSET
        else:
            rotation = ((3/2)) * np.pi + ROBOT_Q_ROTATION_OFFSET
            offset = -HOOP_LEN

        self.offset = offset
        self.hoop_rotation = rotation

        pose[1] += offset
        # pose[2] += MAZE_OFFSET
        pose[2] += 0.025
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


    def __select_q_closest(self, current: list, qs: list):
        assert len(qs) > 0, "No Qs"
        best_metric = float("inf") # lower is better
        best_q = None

        for q in qs:
            metric = np.linalg.norm(current - q)
            if metric < best_metric:
                best_metric = metric
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
    
    def get_q_sequence_tostart(self):
        print("Calculating trajectory to start position")

        configurations = []
        start_pose = self.calculate_ee_start_pose()

        # Let the hoop rotate in a safe area
        q_safe = [0.5, 0.5, 0, 0]
        configurations.append(q_safe)
        q_rotated = q_safe.copy()
        q_rotated[3] = start_pose[3]
        configurations.append(q_rotated)

        # With the hoop rotated continue to the start position
        start_q = self.__select_q_min_rotation(self.current, self.calculate_ik(start_pose))
        configurations.append(start_q)

        self.last_q = start_q
        self.last_pose = start_pose

        print(f"start pose {start_pose}")

        return configurations

    def get_q_sequence_throughmaze(self):
        print("Calculating trajectory through the maze")
        configurations = []

        for i, point in enumerate(self.maze_points[::-1]):
            pose = self.point_to_pose(point)

            if i == (len(self.maze_points)-1): # last movement stops higher
                pose[2] += 0.02

            print(f"pose no {i}: {pose}")
            if np.isclose(self.last_pose[0], pose[0]) and np.isclose(self.last_pose[1], pose[1]): # vertical segment
                print(f"vertical segment for i={i}")
                # if i == 0 or i == 1: # first segments
                qs_height_only = self.__select_qs_height_change_only(self.last_q, self.calculate_ik(pose))
                q = self.__select_q_min_rotation(self.last_q, qs_height_only)
                configurations.append(q)
            else:
                q = self.__select_q_closest(self.last_q, self.calculate_ik(pose))
                configurations.append(q)

            self.last_pose = pose
            self.last_q = q
        
        return configurations


    
    def maze_to_robot_points(self):
        # Transformation matrix
        translation_SE3 = SE3(translation=self.board_center[0:3])
        rotation_SO3 = SO3.rz(self.board_rotation)
        rotation_SE3 = SE3(rotation=rotation_SO3)

        transform = translation_SE3 * rotation_SE3

        # Transform maze points to robot points
        maze_points = MAZE_SHAPES[self.maze_type]
        robot_points = np.empty(maze_points.shape)
        for i, point in enumerate(maze_points):
            transformed_point = transform.act(point)
            robot_points[i] = transformed_point

        return robot_points

    def point_to_pose(self, point):
        pose = point.tolist()
        pose.append(0)

        pose[1] += self.offset
        pose[3] = self.hoop_rotation

        return pose

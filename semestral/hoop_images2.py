from ctu_bosch_sr450 import RobotBosch
from camera import CameraHelper
from utils import LEFT_Q_EXTREME, LEFT_Q_MID, RIGHT_Q_EXTREME, RIGHT_Q_MID, MIN_Z_Q, MAX_Z_Q
import math
import numpy as np
import cv2

DEFAULT_INCREMENT = 0.1
DEBUG = True

def save_matrix(name, matrix):
    np.save("./calibration_data/" + name, matrix)

def rotate(r: RobotBosch, increment = 0.1):
    q = r.get_q()
    q[3] = (q[3] + increment) % (2 * np.pi)
    print("moving to q", q)
    r.move_to_q(q)
    r.wait_for_motion_stop()

def shift_height(r: RobotBosch, new_h_q: float):
    print("Shifting height")
    q = r.get_q()
    q[2] = new_h_q
    r.move_to_q(q)
    r.wait_for_motion_stop()

def find_center(img: list) -> list | None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=300, param1=100, param2=30, minRadius=50, maxRadius=400)

    if circles is None:
        print("Could not find circles")
        return None

    circles = np.uint16(np.around(circles))

    return circles[0][0]

def get_data_in_pose(r: RobotBosch, camera: CameraHelper, pose_name: str):
    print("Getting data")
    for i in range(math.floor((np.pi * 2) / DEFAULT_INCREMENT)):
        rotate(r, DEFAULT_INCREMENT)
        img = camera.get_image()
        center = find_center(img)

        if center is None:
                continue

        print("Found circle: ", center)

        if DEBUG:
            copy = img.copy()
            cv2.circle(copy, center=(center[0], center[1]), radius=50, color=(0, 0, 255)) 
            im_small = cv2.resize(copy, (500, 500))
            cv2.imshow("output", im_small)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        inputk = input("Save? [y]")

        if DEBUG == False or inputk == "":
            pose = list(r.fk(r.get_q()))
            save_matrix("robot_pose_" + pose_name + str(i), pose)
            save_matrix("circle_coords_" + pose_name + str(i), center)


def rotate_to_zero(r: RobotBosch):
    q = r.get_q()
    q[3] = 0
    r.move_to_q(q)
    r.wait_for_motion_stop()

robot = RobotBosch()
robot.initialize()
camera = CameraHelper(robot, "")

q_positions = [RIGHT_Q_EXTREME, RIGHT_Q_MID, LEFT_Q_EXTREME, LEFT_Q_MID]

pose_names = ["RE", "RM", "LE", "LM"]

for i, pos in enumerate(q_positions):
    rotate_to_zero(robot)
    
    robot.move_to_q(pos)
    robot.wait_for_motion_stop()

    shift_height(robot, MAX_Z_Q)
    get_data_in_pose(robot, camera, pose_name[i] + "HIGH_")
    
    rotate_to_zero(robot)

    shift_height(robot, MIN_Z_Q)
    get_data_in_pose(robot, camera, pose_name[i] + "LOW_")

robot.hard_home()
robot.close()

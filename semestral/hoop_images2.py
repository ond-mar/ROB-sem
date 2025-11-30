from ctu_bosch_sr450 import RobotBosch
from camera import CameraHelper
from utils import LEFT_Q, RIGHT_Q, MIN_Z_Q, MAX_Z_Q
import math
import numpy as np
import cv2

DEFAULT_INCREMENT = 0.1
DEBUG = True

def save_matrix(name, matrix):
    np.save("./calibration_data/" + name, matrix)

def rotate(r: RobotBosch, increment = 0.1):
    q = r.get_q()
    q[3] = q[3] + increment
    r.move_to_q(q)
    r.wait_for_motion_stop()

def shift_height(r: RobotBosch, new_h_q: float):
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

def get_data_in_pose(r: RobotBosch, camera: CameraHelper):
    for i in range(math.floor((np.pi * 2) / DEFAULT_INCREMENT)):
        rotate(r, DEFAULT_INCREMENT)
        img = camera.get_image()

        if DEBUG:
            im_small = cv2.resize(img, (960, 540))
            cv2.imshow("output", im_small)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 

        center = find_center(img)

        if center is None:
            return

        pose = list(r.fk(r.get_q()))

        save_matrix("robot_pose_" + str(i), pose)
        save_matrix("circle_coords_" + str(i), center)


robot = RobotBosch()
robot.initialize()
camera = CameraHelper(robot, "")

robot.move_to_q(LEFT_Q)
robot.wait_for_motion_stop()

get_data_in_pose(robot, camera)

shift_height(robot, MIN_Z_Q)

get_data_in_pose(robot, camera)

shift_height(robot, MAX_Z_Q)

r.move_to_q(RIGHT_Q)
r.wait_for_motion_stop()

get_data_in_pose(robot, camera)

shift_height(robot, MIN_Z_Q)

get_data_in_pose(robot, camera)

shift_height(robot, MAX_Z_Q)

robot.hard_home()
robot.close()

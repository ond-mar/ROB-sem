import numpy as np
from camera import CameraHelper
from ctu_bosch_sr450 import RobotBosch
import cv2
import utils

def find_base_rotation(corners, ids, tvec, T_camera2robot):
    tvec = np.asarray(tvec)

    index_id1 = np.where(ids == 1)[0][0]
    index_id2 = np.where(ids == 2)[0][0]
    t_m1_cam = tvec[index_id1][0]
    t_m2_cam = tvec[index_id2][0]

    # Transform to robot coordinates
    t_m1_rob = T_camera2robot @ np.asarray(np.hstack([t_m1_cam, 1]))
    t_m2_rob = T_camera2robot @ np.asarray(np.hstack([t_m2_cam, 1]))

    # Find angle
    t_angle = t_m2_rob - t_m1_rob
    angle = np.atan2(t_angle[1], t_angle[0]) - np.pi/4

    return angle

# Load callibraion
k = np.load("calibration/k.npy")
dist = np.load("calibration/dist.npy")
t_camera2robot = np.load("calibration/camera2robot.npy")

# Robot init
robot = RobotBosch()
robot.initialize(True)
cam = CameraHelper(robot, "lab")

# Hide robot
q_max = robot.q_max.copy()
q_max[3] = 0
robot.move_to_q(q_max)
robot.wait_for_motion_stop()

# Capture img
img = cam.get_image()
img_debug = img.copy()


# Detect ArUcos
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, rejected = detector.detectMarkers(gray)

assert ids is not None and len(ids) == 2, "Wrong number of markers detected"

# Debug image
cv2.aruco.drawDetectedMarkers(img_debug, corners, ids)
cv2.imwrite("debug/rotation.png", img_debug)

rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.04, k, dist)

# Get angle
angle = find_base_rotation(corners, ids, tvec, t_camera2robot)

# Move robot to free space
q_free = [-np.pi/3, np.pi/2, 0, 0]
robot.move_to_q(q_free)
robot.wait_for_motion_stop()
input("press any key")

# Swith angle range
angle_rob = np.fmod(angle, 2*np.pi) + (2*np.pi * (angle < 0))

# Rotate loop in the same direction as labyrinth
q_rot = q_free.copy()
q_rot[3] = angle_rob + utils.ROBOT_Q_ROTATION_OFFSET

robot.move_to_q(q_rot)
robot.wait_for_motion_stop()
input("press any key")

# End of working with robot
robot.soft_home()
robot.close()



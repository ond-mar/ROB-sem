import numpy as np
from camera import CameraHelper
from ctu_bosch_sr450 import RobotBosch
import cv2
import utils

class MazePoseSolver:
    def __init__(self, image, T_camera2robot: np.ndarray, k: np.ndarray, dist: np.ndarray):
        self.image = image
        self.T_camera2robot = T_camera2robot

        # Process image
        img_copy = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        self.ids = ids
        self.corners = corners


        cv2.aruco.drawDetectedMarkers(img_copy, corners, ids) # Export image with ArUcos detected
        cv2.imwrite("debug/table.png", img_copy)

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.04, k, dist)
        self.tvec = np.asarray(tvec)

        assert len(tvec) == 2, "Wrong number of markers detected"

    def find_base_center(self):
        # img_copy = self.image.copy()
        # gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # parameters = cv2.aruco.DetectorParameters()
        # detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        # corners, ids, rejected = detector.detectMarkers(gray)

        # cv2.aruco.drawDetectedMarkers(img_copy, corners, ids) # Export image with ArUcos detected
        # cv2.imwrite("debug/table.png", img_copy)

        # rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.04, self.k, self.dist)
        # tvec = np.asarray(tvec)

        # if len(tvec) < 2:
        #     print("Could not find markers")
        #     return None

        center_cam = (self.tvec[0][0] + self.tvec[1][0]) / 2
        center_cam = np.asarray(np.hstack([center_cam, 1]))
        return list(self.T_camera2robot @ center_cam)
      
    def find_base_rotation(self):
        index_id1 = np.where(self.ids == 1)[0][0]
        index_id2 = np.where(self.ids == 2)[0][0]
        t_m1_cam = self.tvec[index_id1][0]
        t_m2_cam = self.tvec[index_id2][0]

        # Transform to robot coordinates
        t_m1_rob = self.T_camera2robot @ np.asarray(np.hstack([t_m1_cam, 1]))
        t_m2_rob = self.T_camera2robot @ np.asarray(np.hstack([t_m2_cam, 1]))

        # Find angle
        t_angle = t_m2_rob - t_m1_rob
        angle = np.atan2(t_angle[1], t_angle[0]) - np.pi/4

        return angle



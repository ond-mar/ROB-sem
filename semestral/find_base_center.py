import cv2
import numpy as np

def find_base_center(img, debug=True):
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Set up ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    # Detect markers
    corners, ids, rejected_candidates = detector.detectMarkers(img_gray)

    assert ids is not None and len(ids) == 2, "Wrong number of ArUco markers detected"

    # Find base center
    top_left = corners[0][0][0]    
    bottom_right = corners[1][0][2]    

    base_center_x = int((top_left[0] + bottom_right[0]) / 2)
    base_center_y = int((top_left[1] + bottom_right[1]) / 2)

    if debug:
        cv2.circle(img_copy, (int(top_left[0]), int(top_left[1])), 5, (255, 0, 0), -1)
        cv2.circle(img_copy, (int(bottom_right[0]), int(bottom_right[1])), 5, (255, 0, 0), -1)
        cv2.circle(img_copy, (base_center_x, base_center_y), 5, (0, 255, 0), -1)
        cv2.imwrite("debug/base_aruco.png", img_copy)

    return np.array([base_center_x, base_center_y])

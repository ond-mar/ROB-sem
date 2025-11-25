import cv2
from pathlib import Path
import numpy as np

charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((4, 4), 0.025, 0.018, charuco_dict)
board.setLegacyPattern(True)
detector = cv2.aruco.CharucoDetector(board)

DIR_PATH = Path("board_images")

all_corners = []
all_ids = []
image_size = []

for im_path in DIR_PATH.iterdir():
    img = cv2.imread(im_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    
    c_corners, c_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    cv2.aruco.drawDetectedCornersCharuco(img, c_corners, c_ids)
    
    # im_small = cv2.resize(img, (960, 540))
    # cv2.imshow("output", im_small)
    # cv2.waitKey(0) 

    if len(c_corners) > 0:
        image_size.append(size)
        all_ids.append(c_ids)
        all_corners.append(c_corners)


_, k_matrix, distorts, r, t = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, image_size[0], np.zeros((3, 3)), np.zeros((5, 1)))

print(k_matrix, distorts)

np.save("./calibration/k.npy", k_matrix)
np.save("./calibration/dist.npy", distorts)


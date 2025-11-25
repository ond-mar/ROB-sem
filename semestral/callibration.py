import numpy as np
import cv2
from typing import List
from numpy.typing import ArrayLike
from pathlib import Path
from ctu_bosch_sr450 import RobotBosch
from utils import load_matrix

# Find homography method
def find_hoop_homography(images: ArrayLike, hoop_positions: List[dict]) -> np.ndarray:
    images = np.asarray(images)
    assert images.shape[0] == len(hoop_positions)

    img_centers = np.empty((len(hoop_positions), 2))

    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to HSV
        img = cv2.medianBlur(img, 5) # apply median blur to reduce noise
        cv2.waitKey(0)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=300, param1=100, param2=30, minRadius=50, maxRadius=400)

        if circles is None:
            print("Could not find circles")
            continue

        circles = np.uint16(np.around(circles))

        x, y, r = circles[0][0]
        img_centers[i] = [x, y]

        # copy = img.copy()
        # cv2.circle(copy, center=(x, y), radius=2, color=(0, 0, 255))

        # cv2.imshow("image", copy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    ref_centers = np.empty((len(hoop_positions), 2))
    for i, hoop in enumerate(hoop_positions):
        pos = hoop["translation_vector"]
        ref_centers[i] = [pos[0], pos[1]] 

    homography = cv2.findHomography(img_centers, ref_centers, cv2.RANSAC)[0]
    return homography

robot = RobotBosch(tty_dev=None)

DIR_PATH = Path("hoop_images")

stems = [f.stem for f in DIR_PATH.iterdir()]

images = []
vectors = []

for stem in stems:
    mat_path = DIR_PATH / (stem + ".npy")
    im_path = DIR_PATH / (stem + ".png")
    matrix = load_matrix(mat_path)
    image = cv2.imread(im_path)

    coords = list(robot.fk(matrix))

    coords[0] += 0.135 * np.cos(coords[3])
    coords[1] += 0.135 * np.sin(coords[3])

    images.append(image)
    vectors.append({'translation_vector': np.asarray(coords)})


homography = find_hoop_homography(images, vectors)

np.save("homography", homography)
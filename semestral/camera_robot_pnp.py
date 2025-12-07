from ctu_bosch_sr450 import RobotBosch
from pathlib import Path
import cv2
import numpy as np
from typing import List
from numpy.typing import ArrayLike
from utils import load_matrix

def find_centers(images: ArrayLike) -> np.ndarray:
    images = np.asarray(images)

    centers = np.empty((len(images), 2))

    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 5)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=300, param1=100, param2=30, minRadius=50, maxRadius=400)

        if circles is None:
            print("Could not find circles")
            continue

        circles = np.uint16(np.around(circles))

        x, y, r = circles[0][0]
        centers[i] = [x, y]

    return centers

robot = RobotBosch()
robot.initialize(False)

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

    coords[0] += 0.135 * np.cos(coords[3] - np.pi)
    coords[1] += 0.135 * np.sin(coords[3] - np.pi)

    images.append(image)
    vectors.append(coords[0:3])

k = np.load("./calibration/k.npy")
dist = np.load("./calibration/dist.npy")


centers = find_centers(images)

_, r, t = cv2.solvePnP(np.asarray(vectors, dtype = np.float32), np.asarray(centers, dtype = np.float32), np.asarray(k, dtype = np.float32), np.asarray(dist, dtype = np.float32))

transform = np.eye(4)

transform[:3, :3] = cv2.Rodrigues(r)[0]
transform[:3, 3] = t.flatten()

print(t)


inverse = np.linalg.inv(transform)

print(inverse)

np.save("./calibration/camera2robot.npy", inverse)
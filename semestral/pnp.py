from pathlib import Path
import cv2
from utils import load_matrix
import numpy as np


def to_float32(arr):
    return np.asarray(arr, dtype = np.float32)

def load_data():
    DIR_PATH = Path("calibration_data")
    stems = [f.stem for f in DIR_PATH.iterdir()]

    poses = []
    centers = []

    for stem in stems:
        parts = stem.split("_")

        if 'circle' in parts:
            continue

        robot_pose = load_matrix(DIR_PATH / (stem + ".npy"))

        # print("old pose", robot_pose)

        angle = robot_pose[3] + np.pi

        robot_pose[0] += 0.135 * np.cos(angle)
        robot_pose[1] += 0.135 * np.sin(angle)

        # print("new pose", robot_pose)

        circle_path = str.join("_", ["circle", "coords", parts[2], parts[3]]) + ".npy"

        circle_coords = load_matrix(DIR_PATH / circle_path)

        poses.append(to_float32(robot_pose[0:3]))
        centers.append(to_float32([circle_coords[0], circle_coords[1]]))

    return to_float32(poses), to_float32(centers)

k = to_float32(np.load("./calibration/k.npy"))
dist = to_float32(np.load("./calibration/dist.npy"))

poses, centers = load_data()

_, r, t = cv2.solvePnP(poses.astype('float32'), centers.astype('float32'), k.astype('float32'), dist.astype('float32'))

transform = np.eye(4)
transform[:3, :3] = cv2.Rodrigues(r)[0]
transform[:3, 3] = t.flatten()

np.save("./calibration/robot2camera.npy", transform)

transform = np.linalg.inv(transform)

np.save("./calibration/camera2robot.npy", transform)
from pathlib import Path
from ctu_bosch_sr450 import RobotBosch
import cv2
from datetime import datetime
import os

class CameraHelper:
    def __init__(self, robot: RobotBosch, prefix: str):
        self.prefix = prefix
        self.robot = robot

    def capture_image(self, name: str, save: bool = True):
        image = self.robot.grab_image()
        cv2.imwrite(name + ".png", image)

    def get_image(self):
        return self.robot.grab_image()
        
    def capture_image_date(self, name: str | None, save: bool = True):
        timestamp = datetime.now().isoformat()
        image = self.robot.grab_image()
        cv2.imwrite(self.prefix + "_" + timestamp + ".png", image)

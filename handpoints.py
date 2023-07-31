from __future__ import annotations
import os

import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark
from dotenv import load_dotenv

load_dotenv()


class Point:
    def __init__(self, x: float | int, y: float | int):
        if x > int(os.environ["VIDEO_WIDTH"]):
            x = int(os.environ["VIDEO_WIDTH"])
        if y > int(os.environ["VIDEO_HEIGHT"]):
            y = int(os.environ["VIDEO_HEIGHT"])

        if x is None:
            x = 0.0
        if y is None:
            y = 0.0

        self.x = x
        self.y = y

    def to_tuple(self):
        return self.x, self.y


class HandPoints:
    def __init__(self, landmarks: list[NormalizedLandmark | Point]):
        if isinstance(landmarks[0], NormalizedLandmark):
            self.landmarks = [
                Point(landmark.x * int(os.environ["VIDEO_WIDTH"]), landmark.y * int(os.environ["VIDEO_HEIGHT"]))
                for landmark in landmarks
            ]
        else:
            self.landmarks = landmarks

    def _to_relative(self):
        return HandPoints([Point(point.x - self.wrist.x, point.y - self.wrist.y) for point in self.landmarks])

    def _normalize(self):
        max_point_x = max([abs(point.x) for point in self.landmarks])
        max_point_y = max([abs(point.y) for point in self.landmarks])

        return HandPoints([Point(point.x / max_point_x, point.y / max_point_y) for point in self.landmarks])

    def to_numpy(self):
        return np.array([point.to_tuple() for point in self.landmarks])

    @property
    def wrist(self):
        return self.landmarks[0]

    @property
    def thumb_1(self):
        return self.landmarks[1]

    @property
    def thumb_2(self):
        return self.landmarks[2]

    @property
    def thumb_3(self):
        return self.landmarks[3]

    @property
    def thumb_4(self):
        return self.landmarks[4]

    @property
    def index_finger1(self):
        return self.landmarks[5]

    @property
    def index_finger2(self):
        return self.landmarks[6]

    @property
    def index_finger3(self):
        return self.landmarks[7]

    @property
    def index_finger4(self):
        return self.landmarks[8]

    @property
    def middle_finger1(self):
        return self.landmarks[9]

    @property
    def middle_finger2(self):
        return self.landmarks[10]

    @property
    def middle_finger3(self):
        return self.landmarks[11]

    @property
    def middle_finger4(self):
        return self.landmarks[12]

    @property
    def ring_finger1(self):
        return self.landmarks[13]

    @property
    def ring_finger2(self):
        return self.landmarks[14]

    @property
    def ring_finger3(self):
        return self.landmarks[15]

    @property
    def ring_finger4(self):
        return self.landmarks[16]

    @property
    def pinkey_finger1(self):
        return self.landmarks[17]

    @property
    def pinkey_finger2(self):
        return self.landmarks[18]

    @property
    def pinkey_finger3(self):
        return self.landmarks[19]

    @property
    def pinkey_finger4(self):
        return self.landmarks[20]

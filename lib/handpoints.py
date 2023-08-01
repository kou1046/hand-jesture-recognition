from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import cv2
from dacite import from_dict
import ndjson


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __post_init__(self):
        if self.x is None:
            object.__setattr__("x", 0.0)
        if self.y is None:
            object.__setattr__("y", 0.0)

    def to_tuple(self):
        return self.x, self.y

    def to_inttuple(self):
        return int(self.x), int(self.y)


@dataclass(frozen=True)
class HandPoints:
    values: list[Point]

    def _to_relative(self):
        return HandPoints([Point(point.x - self.wrist.x, point.y - self.wrist.y) for point in self.values])

    def _normalize(self):
        max_point_x = max([abs(point.x) for point in self.values])
        max_point_y = max([abs(point.y) for point in self.values])

        return HandPoints([Point(point.x / max_point_x, point.y / max_point_y) for point in self.values])

    def to_numpy(self):
        return np.array([point.to_tuple() for point in self.values])

    def draw(self, image: np.ndarray):
        # thumb
        cv2.line(image, self.thumb_2.to_inttuple(), self.thumb_3.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.thumb_2.to_inttuple(), self.thumb_3.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.thumb_3.to_inttuple(), self.thumb_4.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.thumb_3.to_inttuple(), self.thumb_4.to_inttuple(), (255, 255, 255), 2)

        # index_finger
        cv2.line(image, self.index_finger1.to_inttuple(), self.index_finger2.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.index_finger1.to_inttuple(), self.index_finger2.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.index_finger2.to_inttuple(), self.index_finger3.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.index_finger2.to_inttuple(), self.index_finger3.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.index_finger3.to_inttuple(), self.index_finger4.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.index_finger3.to_inttuple(), self.index_finger4.to_inttuple(), (255, 255, 255), 2)

        # middle finter
        cv2.line(image, self.middle_finger1.to_inttuple(), self.middle_finger2.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.middle_finger1.to_inttuple(), self.middle_finger2.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.middle_finger2.to_inttuple(), self.middle_finger3.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.middle_finger2.to_inttuple(), self.middle_finger3.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.middle_finger3.to_inttuple(), self.middle_finger4.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.middle_finger3.to_inttuple(), self.middle_finger4.to_inttuple(), (255, 255, 255), 2)

        # ring finger
        cv2.line(image, self.ring_finger1.to_inttuple(), self.ring_finger2.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.ring_finger1.to_inttuple(), self.ring_finger2.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.ring_finger2.to_inttuple(), self.ring_finger3.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.ring_finger2.to_inttuple(), self.ring_finger3.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.ring_finger3.to_inttuple(), self.ring_finger4.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.ring_finger3.to_inttuple(), self.ring_finger4.to_inttuple(), (255, 255, 255), 2)

        # # pinkey_finter
        cv2.line(image, self.pinkey_finger1.to_inttuple(), self.pinkey_finger2.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.pinkey_finger1.to_inttuple(), self.pinkey_finger2.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.pinkey_finger2.to_inttuple(), self.pinkey_finger3.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.pinkey_finger2.to_inttuple(), self.pinkey_finger3.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.pinkey_finger3.to_inttuple(), self.pinkey_finger4.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.pinkey_finger3.to_inttuple(), self.pinkey_finger4.to_inttuple(), (255, 255, 255), 2)

        # palm
        cv2.line(image, self.wrist.to_inttuple(), self.thumb_1.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.wrist.to_inttuple(), self.thumb_1.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.thumb_1.to_inttuple(), self.thumb_2.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.thumb_1.to_inttuple(), self.thumb_2.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.thumb_2.to_inttuple(), self.index_finger1.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.thumb_2.to_inttuple(), self.index_finger1.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.index_finger1.to_inttuple(), self.middle_finger1.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.index_finger1.to_inttuple(), self.middle_finger1.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.middle_finger1.to_inttuple(), self.ring_finger1.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.middle_finger1.to_inttuple(), self.ring_finger1.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.ring_finger1.to_inttuple(), self.pinkey_finger1.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.ring_finger1.to_inttuple(), self.pinkey_finger1.to_inttuple(), (255, 255, 255), 2)
        cv2.line(image, self.pinkey_finger1.to_inttuple(), self.wrist.to_inttuple(), (0, 0, 0), 6)
        cv2.line(image, self.pinkey_finger1.to_inttuple(), self.wrist.to_inttuple(), (255, 255, 255), 2)

    @property
    def wrist(self):
        return self.values[0]

    @property
    def thumb_1(self):
        return self.values[1]

    @property
    def thumb_2(self):
        return self.values[2]

    @property
    def thumb_3(self):
        return self.values[3]

    @property
    def thumb_4(self):
        return self.values[4]

    @property
    def index_finger1(self):
        return self.values[5]

    @property
    def index_finger2(self):
        return self.values[6]

    @property
    def index_finger3(self):
        return self.values[7]

    @property
    def index_finger4(self):
        return self.values[8]

    @property
    def middle_finger1(self):
        return self.values[9]

    @property
    def middle_finger2(self):
        return self.values[10]

    @property
    def middle_finger3(self):
        return self.values[11]

    @property
    def middle_finger4(self):
        return self.values[12]

    @property
    def ring_finger1(self):
        return self.values[13]

    @property
    def ring_finger2(self):
        return self.values[14]

    @property
    def ring_finger3(self):
        return self.values[15]

    @property
    def ring_finger4(self):
        return self.values[16]

    @property
    def pinkey_finger1(self):
        return self.values[17]

    @property
    def pinkey_finger2(self):
        return self.values[18]

    @property
    def pinkey_finger3(self):
        return self.values[19]

    @property
    def pinkey_finger4(self):
        return self.values[20]


@dataclass(frozen=True)
class LabeledHandPoints:
    label: int
    handpoints: HandPoints


class LabeledHandPointsStore:
    def save(self, labeled_handpoints: LabeledHandPoints) -> None:
        ...

    def load(self) -> list[LabeledHandPoints]:
        ...


class NdJsonLabeledHandPointsStore(LabeledHandPointsStore):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def save(self, labeled_handpoints: LabeledHandPoints) -> None:
        with open(self.file_path, "a") as f:
            writer = ndjson.writer(f)
            writer.writerow(asdict(labeled_handpoints))

    def load(self) -> list[LabeledHandPoints]:
        with open(self.file_path, "r") as f:
            dataset: list[dict] = ndjson.load(f)
            dataset = [from_dict(LabeledHandPoints, data) for data in dataset]
        return dataset

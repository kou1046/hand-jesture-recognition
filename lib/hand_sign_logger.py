from __future__ import annotations
from enum import Enum
import os

import cv2
import numpy as np

from .hand_detector import HandDetector
from .handpoints import LabeledHandPointsStore, LabeledHandPoints, NdJsonLabeledHandPointsStore


class Mode(Enum):
    DISPLAY = 0
    LOG = 1


class HandSignDataLoggerGUI:
    def __init__(
        self,
        store: LabeledHandPointsStore = NdJsonLabeledHandPointsStore(
            os.path.join("models", "labeled_handpoints.ndjson")
        ),
    ):
        self.store = store
        self.mode: Mode = Mode.DISPLAY
        self.label: int | None = None

    def _display(self, img: np.ndarray):
        display_text = f"{self.mode}" if self.mode == Mode.DISPLAY else f"{self.mode} (label={self.label})"
        text_color = (255, 0, 0) if self.mode == Mode.DISPLAY else (0, 0, 255)

        cv2.putText(img, display_text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 1, cv2.LINE_AA)
        cv2.imshow("img", img)
        key = cv2.waitKey(1)
        if key == 27:  # Esc
            cv2.destroyAllWindows()
            exit()
        if key >= ord("1") and key <= ord("9"):  # 1 ~ 9
            self.mode = Mode.LOG
            self.label = int(chr(key))
        if key == ord("d"):  # d
            self.mode = Mode.DISPLAY

    def start(self):
        detector = HandDetector()
        cap = cv2.VideoCapture(0)

        while True:
            ret, img = cap.read(0)
            handpoints = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if handpoints is None:
                self._display(img)
                continue

            handpoints.draw(img)
            if self.mode == Mode.LOG:
                self.store.save(LabeledHandPoints(self.label, handpoints))

            self._display(img)

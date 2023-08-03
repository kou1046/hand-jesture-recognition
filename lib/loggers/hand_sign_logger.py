from __future__ import annotations
from enum import Enum

import cv2
import numpy as np

from ..detectors.hand_detector import HandDetector
from ..types import LabeledHandPoints
from ..stores import LabeledHandPointsStore


class Mode(Enum):
    DISPLAY = 0
    LOG = 1


class HandSignDataLoggerGUI:
    """
    startで起動すると学習に必要な座標データを記録することが出来る.
    0 - 9 キーを押すと指定したキーの数字ラベルで座標を記録する. dキーを押すとディスプレイモード（記録しない）.
    登録したラベルに飛び番号(0,1,3で登録など)があると学習時にエラーを起こすので注意．
    """

    def __init__(self, store: LabeledHandPointsStore):
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

        if key >= ord("0") and key <= ord("9"):  # 0 ~ 9
            self.mode = Mode.LOG
            label = int(chr(key))
            self._change_label(label)

        if key == ord("d"):  # d
            self.mode = Mode.DISPLAY
            self._change_label(None)

    def _change_label(self, label: int | None):
        self.label = label

    def start(self):
        detector = HandDetector()
        cap = cv2.VideoCapture(0)

        while True:
            ret, img = cap.read()
            handpoints = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if handpoints is None:
                self._display(img)
                continue

            handpoints.draw(img)
            if self.mode == Mode.LOG:
                self.store.save(LabeledHandPoints(self.label, handpoints))

            self._display(img)

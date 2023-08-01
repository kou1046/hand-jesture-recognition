import mediapipe as mp
import numpy as np

from .handpoints import HandPoints, Point


class HandDetector:
    def __init__(self, **kwargs):
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, **kwargs)

    def detect(self, image: np.ndarray):
        height, width, _ = image.shape
        result = self.hands.process(image)

        if result.multi_hand_landmarks is None:
            return None

        landmark = result.multi_hand_landmarks[0].landmark

        # デフォルトの出力はx, yがスケーリングされているので絶対座標に復元する
        handpoints = HandPoints([Point(lm.x * width, lm.y * height) for lm in landmark])

        return handpoints

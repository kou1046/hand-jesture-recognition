import os

import cv2
import numpy as np

from handpoints import HandPoints
from hand_detector import HandDetector

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.environ["VIDEO_WIDTH"]))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.environ["VIDEO_HEIGHT"]))

detector = HandDetector()


while True:
    _, img = cap.read()
    handpoints = detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if handpoints is not None:
        print(handpoints._to_relative()._normalize().to_numpy())

    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

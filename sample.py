import os

import cv2
import mediapipe as mp
import numpy as np

from handpoints import HandPoints
from dotenv import load_dotenv

load_dotenv()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.environ["VIDEO_WIDTH"]))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.environ["VIDEO_HEIGHT"]))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    _, img = cap.read()
    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        handpoints = HandPoints(result.multi_hand_landmarks[0].landmark)

        print(handpoints._to_relative()._normalize().to_numpy())

    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

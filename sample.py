import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
    _, img = cap.read()
    result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for i, lm in enumerate(hand_landmarks.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.putText(
                    img,
                    str(i + 1),
                    (cx + 10, cy + 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    4,
                    (255, 255, 255),
                    5,
                    cv2.LINE_AA,
                )
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

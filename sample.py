import os
import cv2

from lib import HandDetector, HandSignClassifier

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    classifier = HandSignClassifier(2, pretrained_model_path=os.path.join("models", "hand_sign_classifier_weights.pth"))

    while True:
        ret, frame = cap.read()
        handpoints = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if handpoints is not None:
            handpoints.draw(frame)
            sign = classifier.predict(handpoints)
            print(sign)

        cv2.imshow("img", frame)
        cv2.waitKey(1)

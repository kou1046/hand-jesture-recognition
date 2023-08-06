import cv2
import sys

sys.path.append(".")
from lib import HandDetector, HandSignClassifier
from examples.train_classifier import SAVE_MODEL_PATH, HandSign

cap = cv2.VideoCapture(0)
detector = HandDetector()
classifier = HandSignClassifier(output_size=len(HandSign), pretrained_model_path=SAVE_MODEL_PATH)

while True:
    ret, frame = cap.read()
    handpoints = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if handpoints is not None:
        handpoints.draw(frame)
        sign_label = classifier.predict(handpoints)
        handsign_name = HandSign(sign_label).name
        xmin, ymin, xmax, ymax = handpoints.bbox()
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
        cv2.putText(frame, handsign_name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.imshow("img", frame)
    cv2.waitKey(1)

import os 
from dataclasses import asdict

import ndjson
import dacite
import cv2

from classifier import HandSignClassifier
from lib import LabeledHandPointsStore
from lib.handpoints import LabeledHandPoints
from lib import HandDetector


if __name__ == "__main__":
    
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    classifier = HandSignClassifier(pretrained_model_path=os.path.join("models", "hand_sign_classifier.pth"))
    
    while True:
        ret, frame = cap.read()
        handpoints = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if handpoints is not None:
            handpoints.draw(frame)
            sign = classifier.predict(handpoints)
            print(sign)
            
            
        cv2.imshow("img", frame)
        cv2.waitKey(1)
        
    
    
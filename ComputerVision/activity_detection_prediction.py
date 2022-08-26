import cv2
import mediapipe as mp
from feature_extraction import extract_features
import pickle
import requests
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
activity_detector = pickle.load(
    open('./models/activity_detection_v2.pkl', 'rb'))


"""  # Visualization of Pose Estimation and Activity Detection"""


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Initialize probability list
        probas, mean_proba, proba_counter = [], -1, 0
        activity = ''
        while cap.isOpened():
            ret, frame = cap.read()

            try:
                # Extract Features
                features = extract_features(frame)
                # Predict the activity proba and append to the list
                y = activity_detector.predict_proba(features)[0][1]
                probas.append(y)
                proba_counter += 1
                # Aggregate prediction results and decide activity for the time frame
                if proba_counter == 50:
                    mean_proba = np.mean(probas)
                    if mean_proba >= 0.5:
                        activity = 'Working'
                        requests.get('http://'+os.environ['LS_HOST']+'/events/trigger?eventID=1')
                    else:
                        activity = 'Not Working'
                        requests.get('http://'+os.environ['LS_HOST']+'/events/trigger?eventID=0')
                    probas = []
                    proba_counter = 0
                    print('Aggregated Result:', activity)

                print('Working Proba:', y)

            except Exception as e:
                print(str(e))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

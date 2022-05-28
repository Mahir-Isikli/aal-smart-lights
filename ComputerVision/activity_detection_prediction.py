import cv2
import mediapipe as mp
from feature_extraction import estimate_pose, calculate_all_angles
from visualization import draw_landmarks, draw_text
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
activity_detector = pickle.load(open('activity_detection_model.pkl', 'rb'))


"""  # Visualization of Pose Estimation and Activity Detection"""


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make pose detections
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Render pose detections
            draw_landmarks(image, results.pose_landmarks)

            try:
                # Calculate angles
                features = calculate_all_angles(results.pose_landmarks.landmark)
                # Predict the activity
                y = activity_detector.predict(features)[0]
                # Render activity detection
                if y > 0.5 :
                    draw_text(image, 'Working')
                    print('Working')
                else:
                    draw_text(image, 'Not Working')
                    print('Not Working')
            except Exception as e:
                print(str(e))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

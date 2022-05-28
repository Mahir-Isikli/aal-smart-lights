"""## Import Requirements"""

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from glob import glob
import gc
import os
# from imblearn.over_sampling import SMOTE

"""## Feature Extraction

### Pose Estimation Model and Transformation Functions
"""

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def estimate_pose(image):
    # Setup mediapipe instance
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    # Recolor image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # # Recolor back to BGR
    # image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # # Render detections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                             mp_drawing.DrawingSpec(
    #                                 color=(245, 117, 66), thickness=2, circle_radius=2),
    #                             mp_drawing.DrawingSpec(
    #                                 color=(245, 66, 230), thickness=2, circle_radius=2)
    #                             )
    # # Return output
    # cv2.imwrite('output.png', image)
    try:
        return results.pose_landmarks.landmark
    except AttributeError:
        return None


def calculate_angle(first, mid, end):
    first = np.array(first)
    mid = np.array(mid)  # Mid
    end = np.array(end)  # End

    radians = np.arctan2(end[1]-mid[1], end[0]-mid[0]) - \
        np.arctan2(first[1]-mid[1], first[0]-mid[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def calculate_all_angles(landmarks):
    angle_map = {
        'SHOULDER': ['ELBOW', 'SHOULDER', 'HIP'],
        'ELBOW': ['SHOULDER', 'ELBOW', 'WRIST'],
        'HIP': ['KNEE', 'HIP', 'SHOULDER'],
        'KNEE': ['HIP', 'KNEE', 'ANKLE'],
        'ANKLE': ['KNEE', 'ANKLE', 'PINKY']
    }

    angles = {}
    for angle_name, landmark_points in angle_map.items():
        for side in ['LEFT', 'RIGHT']:
            # get landmark names
            landmark_names = []
            for point in landmark_points:
                landmark_names.append(side+'_'+point)
            # calculate angle
            first = [landmarks[getattr(mp_pose.PoseLandmark, landmark_names[0]).value].x,
                     landmarks[getattr(mp_pose.PoseLandmark, landmark_names[0]).value].y]
            mid = [landmarks[getattr(mp_pose.PoseLandmark, landmark_names[1]).value].x,
                   landmarks[getattr(mp_pose.PoseLandmark, landmark_names[1]).value].y]
            end = [landmarks[getattr(mp_pose.PoseLandmark, landmark_names[2]).value].x,
                   landmarks[getattr(mp_pose.PoseLandmark, landmark_names[2]).value].y]
            angle = calculate_angle(first, mid, end)
            angles[side+'_'+angle_name] = [angle]

            # cv2.putText(image, str(angle),
            #             tuple(np.multiply(mid, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #             )
            # cv2.imwrite('output.png', image)
    angles = pd.DataFrame(angles)
    return angles


def process_folder(folder_path, name, label, start_idx=0):
    # for each image in the folder
    img_paths = glob(folder_path+'/*.jpg')
    img_paths = img_paths[start_idx:]
    n_process = 10
    n = len(img_paths)
    steps = int(n / n_process)
    final = n % n_process
    for i in range(steps):
        data = pd.DataFrame()
        iter_paths = img_paths[i*n_process:(i+1)*n_process]
        for j, img_path in enumerate(iter_paths):
            print(start_idx+i*n_process+j, img_path)
            # read image
            image = cv2.imread(img_path)
            # estimate pose landmarks
            landmarks = estimate_pose(image)
            del image
            if landmarks:
                # calculate all angles
                angles = calculate_all_angles(landmarks)
                # append to data
                data = pd.concat([data, angles])
        data['label'] = label
        data.to_excel(os.path.join('feature-extraction',
                      name+str(i)+'.xlsx'), index=False)
        del data
        gc.collect()
        print(f'Iter {i} completed!')

if __name__ == '__main__':
    """### Feature Extraction Job for Activity Detection """

    process_folder('activity-detection-data/Not working', 'not-working', 0, start_idx=0)

    process_folder('activity-detection-data/Working', 'working', 1, start_idx=0)

    df = pd.DataFrame()
    for p in glob('feature-extraction/*.xlsx'):
        df = pd.concat([df, pd.read_excel(p).reset_index(drop=True)])
    df.to_excel('working-detection.xlsx', index=False)

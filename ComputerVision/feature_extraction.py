"""## Import Requirements"""

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from glob import glob
import gc
import os
import tensorflow as tf
import tensorflow_hub as hub

# ### Feature Extraction Functions
# Get mediapipe pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Get tf object detection model
object_detection_model = hub.load(
    'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1')

### Pose Estimation ###


def estimate_pose(image):
    # Setup mediapipe instance
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    # Recolor image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

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
        'ANKLE': ['KNEE', 'ANKLE', 'FOOT_INDEX']
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
            visibility = np.mean([landmarks[getattr(mp_pose.PoseLandmark, landmark_names[0]).value].visibility,
                                  landmarks[getattr(
                                      mp_pose.PoseLandmark, landmark_names[1]).value].visibility,
                                  landmarks[getattr(mp_pose.PoseLandmark, landmark_names[2]).value].visibility])
            angle = calculate_angle(first, mid, end)
            angles[side+'_'+angle_name] = [angle]
            angles[side+'_'+angle_name+'_visibility'] = [visibility]

    angles = pd.DataFrame(angles)
    return angles


### Object Detection ###

def resize_image(image, dsize=(640, 640)):
    return cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_CUBIC)


def detect_objects(image, model=object_detection_model):
  if image.shape != (640, 640):
    # Format for the Tensor
    image = resize_image(image)

  # To Tensor
  image_tensor = tf.image.convert_image_dtype(image, tf.uint8)[tf.newaxis, ...]
  # Make detections
  detections = object_detection_model(image_tensor)
  detections = {key: value.numpy() for key, value in detections.items()}
  # Format results as dataframe
  df_result = pd.DataFrame({
      'class': detections['detection_classes'][0],
      'detection_score': detections['detection_scores'][0],
      'ymin': map(lambda x: x[0], detections['detection_boxes'][0]),
      'xmin': map(lambda x: x[1], detections['detection_boxes'][0]),
      'ymax': map(lambda x: x[2], detections['detection_boxes'][0]),
      'xmax': map(lambda x: x[3], detections['detection_boxes'][0]),
  })
  # Filter necessary objects
  objects = {'laptop': df_result[df_result['class'] == 73],
             'keyboard': df_result[df_result['class'] == 76],
             'cellphone': df_result[df_result['class'] == 77], }

  return objects

### Feature Extraction ###


def sightline_intersects(ear, nose, obj_xmin, obj_ymin, obj_xmax, obj_ymax, img_shape):
  sightline = (nose[0]-ear[0], nose[1]-ear[1])
  current_point = (nose[0], nose[1])
  intersects = False
  while current_point[0] < img_shape[0] and current_point[1] < img_shape[1] \
          and current_point[0] > 0 and current_point[0] > 0 and not intersects:
      if current_point[0] < obj_xmax and current_point[1] < obj_ymax \
              and current_point[0] > obj_xmin and current_point[0] > obj_ymin:
          intersects = True
      else:
          current_point = (
              current_point[0] + sightline[0], current_point[1] + sightline[1])
  return intersects


def looks_at(image):
  looks_at_ = {'laptop': 0, 'keyboard': 0, 'cellphone': 0}
  # resize
  image = resize_image(image)
  # estimate pose landmarks
  landmarks = estimate_pose(image)
  if landmarks:
    # detect objects
    objects = detect_objects(image)
    # find objects user is looking at
    nose = landmarks[getattr(mp_pose.PoseLandmark, 'NOSE').value]
    sides = ['LEFT_', 'RIGHT_']
    for side in sides:
      ear = landmarks[getattr(mp_pose.PoseLandmark, side+'EAR').value]
      for obj, df in objects.items():
        for i in df.index:
          obj_row = df.loc[i]
          looks_at_[obj] = int((looks_at_[obj] == True) | sightline_intersects(
              [ear.x, ear.y], [nose.x, nose.y], obj_row.xmin, obj_row.ymin, obj_row.xmax, obj_row.ymax, image.shape))
  return pd.DataFrame({key: [value] for key, value in looks_at_.items()})


def hand_at(image):
  hand_at_ = {'laptop': 0, 'keyboard': 0, 'cellphone': 0}
  # resize
  image = resize_image(image)
  # estimate pose landmarks
  landmarks = estimate_pose(image)
  if landmarks:
    # detect objects
    objects = detect_objects(image)
    # find objects at hand
    fingers = ['LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX']
    for finger in fingers:
      finger = landmarks[getattr(mp_pose.PoseLandmark, finger).value]
      for obj, df in objects.items():
        for i in df.index:
          obj_row = df.loc[i]
          hand_at_[obj] = int((hand_at_[obj] == True) | (obj_row.xmin < finger.x and finger.x < obj_row.xmax and
                                                         obj_row.ymin < finger.y and finger.y < obj_row.ymax))
  return pd.DataFrame({key: [value] for key, value in hand_at_.items()})


def focus_objects(image):
  looks_at_ = {'laptop': 0, 'keyboard': 0, 'cellphone': 0}
  hand_at_ = {'laptop': 0, 'keyboard': 0, 'cellphone': 0}
  # resize
  image = resize_image(image)
  # estimate pose landmarks
  landmarks = estimate_pose(image)
  if landmarks:
    # detect objects
    objects = detect_objects(image)
    # iterate over objects
    for obj, df in objects.items():
      for i in df.index:
        obj_row = df.loc[i]

        # find objects at hand
        fingers = ['LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX']
        for finger in fingers:
          finger = landmarks[getattr(mp_pose.PoseLandmark, finger).value]
          hand_at_[obj] = int((hand_at_[obj] == True) | (obj_row.xmin < finger.x and finger.x < obj_row.xmax and
                                                         obj_row.ymin < finger.y and finger.y < obj_row.ymax))

        # find objects user is looking at
        nose = landmarks[getattr(mp_pose.PoseLandmark, 'NOSE').value]
        sides = ['LEFT_', 'RIGHT_']
        for side in sides:
          ear = landmarks[getattr(mp_pose.PoseLandmark, side+'EAR').value]
          looks_at_[obj] = int((looks_at_[obj] == True) | sightline_intersects(
              [ear.x, ear.y], [nose.x, nose.y], obj_row.xmin, obj_row.ymin, obj_row.xmax, obj_row.ymax, image.shape))

  return pd.concat([pd.DataFrame({'hand_at_'+key: [value] for key, value in hand_at_.items()}),
                    pd.DataFrame({'looks_at_'+key: [value] for key, value in looks_at_.items()})], axis=1)


def extract_features(image, extract_focus_objects=True):
  features = None
  # estimate pose landmarks
  landmarks = estimate_pose(image)
  if landmarks:
    # calculate all angles and assaign to features
    features = calculate_all_angles(landmarks)
    if extract_focus_objects:
      # extract focus objects and concat with features
      features = pd.concat([features, focus_objects(image)], axis=1)
  return features


def process_folder(folder_path, name, label, start_iter=0):
    # for each image in the folder
    img_paths = glob(folder_path+'/*.JPG') + glob(folder_path+'/*.jpg') + \
        glob(folder_path+'/*.png') + glob(folder_path+'/*.PNG')
    n_process = 10
    n = len(img_paths)
    steps = int(n / n_process)
    for i in range(start_iter, steps):
        data = pd.DataFrame()
        iter_paths = img_paths[i*n_process:(i+1)*n_process]
        for j, img_path in enumerate(iter_paths):
            print(i*n_process+j, img_path)
            # read image
            image = cv2.imread(img_path)
            # extract features
            features = extract_features(image)
            del image
            gc.collect()
            data = pd.concat([data, features])
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

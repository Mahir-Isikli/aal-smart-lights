import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


"""  # Visualization of Pose Estimation and Activity Detection"""


def draw_landmarks(image, landmarks):
    mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def draw_text(image, text):
    font = cv2.FONT_HERSHEY_PLAIN
    pos = (25, 25)
    font_scale = 1
    font_thickness = 1
    text_color = (0, 255, 0)
    text_color_bg = (0, 0, 0)

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(image, (x, y-10), (x + text_w, y +
                  text_h + 20), text_color_bg, -1)
    cv2.putText(image, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness)

    return text_size

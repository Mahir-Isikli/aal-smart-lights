import cv2
from matplotlib import pyplot as plt

# Haarcascade files
haarcascade_files = {
    'body': 'haarcascade_fullbody.xml',
    'upperbody': 'haarcascade_upperbody.xml',
    'face2': 'haarcascade_frontalface_alt2.xml'
}

# Load the cascades
haarcascades = {}
for object_name, file_name in haarcascade_files.items():
    haarcascades[object_name] = cv2.CascadeClassifier(
        cv2.data.haarcascades + file_name)

def detect_objects(img, object_names):
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for name in object_names:
        # Detect objects
        objects = haarcascades[name].detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the bodies
        for (x, y, w, h) in objects:
            print(x, y, w, h)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    ## Setup mediapipe instance
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        obj_img = detect_objects(image, ['face2'])

        # Display output
        # plt.imshow(obj_img, cmap = 'gray', interpolation = 'bicubic')
        # plt.xticks([]), plt.yticks([]) 
        # plt.show()
        cv2.imshow('img', obj_img)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

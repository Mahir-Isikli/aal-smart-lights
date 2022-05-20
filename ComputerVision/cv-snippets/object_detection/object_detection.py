import cv2

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
    # Display the output
    cv2.imwrite('output.jpg', img)
    cv2.imshow('img', img)
    

if __name__ == '__main__':
    img = cv2.imread('images/2.jpeg')
    detect_objects(img, haarcascades.keys())

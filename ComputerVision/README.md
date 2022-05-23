# Smart Lights Computer Vision
## Getting Started
1. Clone the repository: git clone https://github.com/Mahir-Isikli/aal-smart-lights.git
2. Go to the project folder: cd aal-smart-lights
3. Switch to branch: git checkout computer-vision
4. Go to the computer vision folder: cd ComputerVision
5. Install requirements: pip3 install -r requirements.txt
6. Install OpenCV:
<br/>If you already have an OpenCV version on your computer create a virtual environment or delete the old version.
We should do it, because OpenCV packages are incompatiable with eachother, if they have different versions.
- pip3 uninstall opencv-python opencv-python-headless opencv-contrib-python
<br/>Then you should install OpenCV packages with pip3:
- pip3 install opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64 opencv-contrib-python==4.5.5.64
7. Run: python3 activity_detection_smartlight.py or jupter notebook activity_detection_smartlight.ipynb

---
#### Resources
1. https://github.com/nicknochnack/MediaPipePoseEstimation

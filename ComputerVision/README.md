# Smart Lights Computer Vision
## Getting Started
1. Clone the repository: git clone https://github.com/Mahir-Isikli/aal-smart-lights.git
2. Go to the project folder: cd aal-smart-lights
3. Go to the computer vision folder: cd ComputerVision
4. Make directories with the names "activity-detection-data" and "feature-extraction": mkdir activity-detection-data feature-extraction
5. Put "Working" and "Not working" folders in the "activity-detection-data" folder.
6. Install the requirements: pip3 install -r requirements.txt
7. Install OpenCV:
<br/>If you already have an OpenCV version on your computer create a virtual environment or delete the old version.
We should do it, because OpenCV packages are incompatiable with eachother, if they have different versions.
- pip3 uninstall opencv-python opencv-python-headless opencv-contrib-python
<br/>Then you should install OpenCV packages with pip3:
- pip3 install opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64 opencv-contrib-python==4.5.5.64
8. Run: python3 activity_detection_modeling.py or jupyter notebook activity_detection_modeling.ipynb

---
#### Resources
1. https://github.com/nicknochnack/MediaPipePoseEstimation

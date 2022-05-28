# Activity Detection
## Getting Started
1. Clone the repository: git clone https://github.com/egeatmaca/activity-detection.git
2. Go to the project folder: cd activity-detection
3. Make directories with the names "activity-detection-data" and "feature-extraction" (only required for training a new model): mkdir activity-detection-data feature-extraction 
4. Put "Working" and "Not working" folders in the "activity-detection-data" folder. (only required for training a new model)
5. Install the requirements: pip3 install -r requirements.txt
6. Install OpenCV:
<br/>If you already have an OpenCV version on your computer create a virtual environment or delete the old version.
We should do it, because OpenCV packages are incompatiable with eachother, if they have different versions.
- pip3 uninstall opencv-python opencv-python-headless opencv-contrib-python
<br/>Then you should install OpenCV packages with pip3:
- pip3 install opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64 opencv-contrib-python==4.5.5.64
7. Run: 
<br/> for feature extraction: python3 feature_extraction.py
<br/> for data prep., model tuning, training: python3 activity_detection_modeling.py or jupyter notebook activity_detection_modeling.ipynb
<br/> for prediction: python3 activity_detection_prediction.py
---

Run real-time prediction with webcam and docker:
1. Build: docker build -t activity-detection .
2. Run: docker run --pid=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro --privileged --device=/dev/video0 activity-detection

---
#### Resources
1. https://github.com/nicknochnack/MediaPipePoseEstimation

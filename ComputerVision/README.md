# Activity Detection
## Getting Started
1. Clone the repository: <pre><code>git clone https://github.com/egeatmaca/activity-detection.git</code></pre>
2. Go to the project folder: <pre><code> cd activity-detection </code></pre>
3. Make directories with the names "activity-detection-data" and "feature-extraction" (only required for training a new model): <pre><code> mkdir activity-detection-data feature-extraction  </code></pre> 
4. Put "Working" and "Not working" folders in the "activity-detection-data" folder. (only required for training a new model)
5. Install the requirements: <pre><code> pip3 install -r requirements.txt </code></pre> 
6. Install OpenCV:
<br/>If you already have an OpenCV version on your computer then you either have to create a virtual environment or delete the old version.
This needs to be done because OpenCV packages are incompatiable with eachother, if they have different versions.
- <pre><code> pip3 uninstall opencv-python opencv-python-headless opencv-contrib-python </code></pre> 
<br/>Then you should install OpenCV packages with pip3:
- <pre><code> pip3 install opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64 opencv-contrib-python==4.5.5.64 </code></pre> 
7. Run: 
* for feature extraction: <pre><code>``` python3 feature_extraction.py ```</code></pre> 
* for data prep., model tuning, training: <pre><code>``` python3 activity_detection_modeling.py ```</code></pre> 
 or  jupyter notebook activity_detection_modeling.ipynb 
* for prediction: <pre><code>``` python3 activity_detection_prediction.py ```</code></pre> 

---

## Run real-time prediction with webcam and docker:
1. Build:  <pre><code> docker build -t activity-detection . </code></pre>
2. Add Root to Access Control List: <pre><code> xhost +local:root </code></pre>
3. Run: <pre><code> docker run --pid=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro --privileged --device=/dev/video0 activity-detection </code></pre> 

---
#### Resources
1. https://github.com/nicknochnack/MediaPipePoseEstimation


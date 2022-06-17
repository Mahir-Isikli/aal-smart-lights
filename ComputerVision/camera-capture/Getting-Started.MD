## Getting Started

To use the camera_capture.py script you will need to use a camera that allows a RTSP streaming connection or another VLC supported streaming protocol.

One viable camera, which is also used by us is the TP Link Tapo C200 Camera [Amazon Link](https://www.amazon.de/-/en/TP-Link-Tapo-C200-Camera-Resolution/dp/B07XLML2YS/ref=sr_1_5?crid=WS3JCJULZYTQ&keywords=c200&qid=1654005032&sprefix=c200%2Caps%2C127&sr=8-5&th=1).

You will first need to download the Tapo App on the [Apple App Store](https://apps.apple.com/de/app/tp-link-tapo/id1472718009?l=en) or [Google Play Store](https://play.google.com/store/apps/details?id=com.tplink.iot&hl=en&gl=US), create a Tapo account and set up your device according to the on-screen instructions. Note that this specific camera only supports 2.4 GHz WiFi.

#### Camera Setup - Set Up Login Details

![assets/1-account-setup.png](assets/1-account-setup.png)

#### Camera Setup - Get Camera IP

![assets/2-ip-address.png](assets/2-ip-address.png)


#### Connect to Camera over VLC
When you have both credentials set up, you can now connect to the camera via the following URL over VLC or another streaming application where you have to change the parameters to your own.

``` 
rtsp://USER:PASSWORD@192.168.1.XX:PORT/stream1
```

![assets/VLC-1.png](assets/VLC-1.png)
![assets/VLC-2.png](assets/VLC-2.png)

### Use the Camera with the Camera Capture Script

Install the VLC Python library for Python 3

```
pip3 install python-vlc
```

In the config.py file you can set the following parameters: 

| Parameter | Description                                               |
| ---|-----------------------------------------------------------|
|rtsp_url | URL of the RTSP stream                                    |
|delay_between_images | Delay between images in seconds                           |
|cronjob_repeat_time | Delay between CRON jobs to circumvent errors being thrown |
|picture_height | Height of the picture snapshot                            |
|picture_width | Width of the picture snapshot                             |

Run the code with the following command and the images should start being captured in the capture folder
```
python3 camera_capture.py
```

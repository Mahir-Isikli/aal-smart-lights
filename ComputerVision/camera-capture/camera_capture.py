import time
import vlc
import datetime
import config
import os

#Start a VLC instance
i = vlc.Instance("--vout=dummy") # when running from cron, there is no screen attached
camera = i.media_player_new()
camera.set_mrl(config.rtsp_url)
camera.audio_set_volume(0)
camera.play()

# Prepare the capture folder
write_path = "captures"
if not os.path.exists(write_path):
    os.makedirs(write_path)

print ('waiting for VLC')
time.sleep(10)
print ("staring capture")

starttime = datetime.datetime.now().timestamp()
while datetime.datetime.now().timestamp() < starttime + config.cronjob_repeat_time:
    cycle_starttime = datetime.datetime.now().timestamp()
    file_name =  "captures/" + datetime.datetime.now().ctime() + ".png"
    print(file_name)
    camera.video_take_snapshot(0,file_name,config.picture_width,config.picture_height)
    while datetime.datetime.now().timestamp() < cycle_starttime + config.delay_between_images:
        time.sleep(0.1)

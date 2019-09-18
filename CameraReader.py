"""
auther @Eng.Elham albaroudi
"""

from VideoGet import VideoGet
from VideoShow import VideoShow
from SaveImage import SaveImage

# path references
#fn = "datasets\parkinglot_1_720p.mp4"
#fn = "datasets\street_high_360p.mp4"
# cascade_src = 'Khare_classifier_02.xml'
# car_cascade = cv2.CascadeClassifier(cascade_src)

dict =  {
        'text_overlay': True,
        'parking_overlay': True,
        'parking_id_overlay': True,
        'parking_detection': True,
        'motion_detection': False,
        'min_area_motion_contour': 500, # area given to detect motion
        'park_laplacian_th': 2.8,
        'park_sec_to_wait': 4, # 4   wait time for changing the status of a region
        'start_frame': 250, # begin frame from specific frame number
        'show_ids': True, # shows id on each region
        'classifier_used': True,
        'save_video': True
        }

def threadBoth(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(source).start()
    #video_shower = VideoShow(video_getter.frame).start()
    save_image = SaveImage(video_getter.video_cur_pos,video_getter.frame_out,video_getter.frame_gray,video_getter.frame)
    save_image.start()
    while True:
        if video_getter.stopped or save_image.stopped:
            #video_shower.stop()
            video_getter.stop()
            save_image.stop()
            #save_image.join()
            break

        save_image.frame = video_getter.frame
        save_image.frame_out = video_getter.frame_out
        save_image.video_cur_pos = video_getter.video_cur_pos
        save_image.frame_gray = video_getter.frame_gray
        #video_shower.frame = save_image.frame_out


if __name__ == '__main__':

    #source ="rtsp://admin:12345678a@192.168.100.219:554/ISAPI/streaming/channels/101"
    source= "camer1.webm"
    threadBoth(source)

"""
auther @Eng.Elham albaroudi

"""

from threading import Thread
import cv2
import time

global_str = "Last change at: "
change_pos = 0.00
start_frame = 300
text_overlay = True

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        # Set from video
        cap = cv2.VideoCapture(src)
        self.video_info = {'fps': cap.get(cv2.CAP_PROP_FPS),
                      'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.6),
                      'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.6),
                      'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
                      'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # jump to frame number specified
        self.stream = cap
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.frame_out = self.frame.copy()
        self.video_cur_pos =     self.stream.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.frame_blur = cv2.GaussianBlur(self.frame.copy(), (5, 5), 3)
        # frame_blur = frame_blur[150:1000, 100:1800]
        self.frame_gray = cv2.cvtColor(self.frame_blur, cv2.COLOR_BGR2GRAY)


    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            self.grabbed, frame_initial = self.stream.read()
            if not self.grabbed:
                self.stop()
            else:
                self.video_cur_pos = self.stream.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current position of the video file in seconds
                video_cur_frame = self.stream.get(cv2.CAP_PROP_POS_FRAMES)  # Index of the frame to be decoded/captured next

                self.frame = cv2.resize(frame_initial, None, fx=0.6, fy=0.6)
                # Background Subtraction
                frame_blur = cv2.GaussianBlur(self.frame.copy(), (5, 5), 3)
                # frame_blur = frame_blur[150:1000, 100:1800]
                self.frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
                self.frame_out = self.frame.copy()

                # Drawing the Overlay. Text overlay at the left corner of screen
                if text_overlay:
                    str_on_frame = "%d/%d" % (video_cur_frame, self.video_info['num_of_frames'])
                    cv2.putText(self.frame_out, str_on_frame, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(self.frame_out, global_str + str(round(change_pos, 2)) + 'sec', (5, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 0, 0), 2, cv2.LINE_AA)
  
    def stop(self):
        self.stopped = True
        self.stream.release()
        cv2.destroyAllWindows()
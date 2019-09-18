"""

auther @Eng.Elham albaroudi
"""


from threading import Thread
import cv2






class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False
        self.counter=0
    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
         while not self.stopped:
            cv2.imshow("Video", self.frame)
            cv2.imwrite('./videos/frameset/%d.jpg' % self.counter, self.frame)
            self.counter+=1
            k = cv2.waitKey(1)
            if k%256 == 27:
               # ESC pressed
               self.stopped = True



    def stop(self):
        self.stopped = True

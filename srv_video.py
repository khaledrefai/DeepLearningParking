# import libraries
from vidgear.gears import NetGear
import cv2

stream = cv2.VideoCapture('camer1.webm')  # Open any video stream



# change following IP address '192.168.1.xxx' with yours
server = NetGear(address='34.66.105.29', port='20', protocol='tcp', pattern=0, receive_mode=False, logging=False)  # Define netgear server at your system IP address.
# change following IP address '192.168.1.xxx' with yours
client = NetGear(address='34.66.105.29', port='20', protocol='tcp', pattern=0, receive_mode=True,
                 logging=True)  # Define netgear client at Server IP address.

# infinite loop until [Ctrl+C] is pressed
while True:
    try:
        (grabbed, frame_initial) = stream.read()
        frame = cv2.resize(frame_initial, None, fx=0.6, fy=0.6)
        # read frames

        # check if frame is not grabbed
        if not grabbed:
            # if True break the infinite loop
            break

        # do something with frame here

        # send frame to server
        server.send(frame)
        frame = client.recv()
    except KeyboardInterrupt:
        # break the infinite loop
        break

# safely close video stream
stream.release()
# safely close server
server.close()
"""A special VideoCamera, suppose to be optimised and threaded using Pyimagesearch's imutils. """
from __future__ import print_function

import logging
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import cv2

# vs = WebcamVideoStream(src=0).start()
# fps = FPS().start()

# loop over some frames...this time using the threaded stream
# while fps._numFrames < args["num_frames"]:
#     # grab the frame from the threaded video stream and resize it
#     # to have a maximum width of 400 pixels
#     frame = vs.read()
#     frame = imutils.resize(frame, width=400)
#
#     # check to see if the frame should be displayed to our screen
#     if args["display"] > 0:
#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1) & 0xFF
#
#     # update the FPS counter
#     fps.update()
#
# # stop the timer and display FPS information
# fps.stop()
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#
# # do a bit of cleanup
# cv2.destroyAllWindows()
# vs.stop()

# Create loggers.
camera_logger = logging.getLogger('camera_handler')
ch = logging.StreamHandler()
# create formatter and add it to the handlers.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to loggers.
camera_logger.addHandler(ch)


class VideoCamera(object):
    def __init__(self):
        """Using OpenCV to capture from device 0. If you have trouble capturing
        from a webcam, comment the line below out and use a video file
        instead.
        """
        self.logger = logging.getLogger('camera_handler')
        self.video = cv2.VideoCapture(0)

        # Threaded attempt for frame streaming, maybe not nessesary.
        # self.video = WebcamVideoStream(src=0).start()

        # Get time of initiation.
        self.fps = FPS().start()

        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if success is True:
            """Update FPS, and incode recieved frame. """
            self.fps.update()
            # TODO: add self.fps.fps() to image, if flagged raised.

            # We are using Motion JPEG, but OpenCV defaults to capture raw images,
            # so we must encode it into JPEG in order to correctly display the
            # video stream.
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            self.logger.debug("in 'get_frame', video.read not success")
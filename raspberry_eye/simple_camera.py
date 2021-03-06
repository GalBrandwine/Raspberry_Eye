"""A simple webcam,

 get video feed from camera, compute FPS,
 and return the capd frame with inframe_printed header "simple camera".
 """

from __future__ import print_function

import logging
# Thought to use WebcamVideoStream from imutils, but there's no need to be threaded
from imutils.video import WebcamVideoStream
from imutils.video import FPS
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
logging.basicConfig(level=logging.INFO)
camera_logger = logging.getLogger('simple_camera')
ch = logging.StreamHandler()
# create formatter and add it to the handlers.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to loggers.
camera_logger.addHandler(ch)


class SimpleCamera:
    def __init__(self, camera_index, logger=None):
        """Using OpenCV to cap from device 0.

        If you have trouble capturing
        from a Web-cam, comment the line below out and use a video file
        instead.
        """
        self.logger = logging.getLogger('simple_camera') if logger is None else logger
        self.camera_index=camera_index
        self.grabbed = None
        self.frame = None
        self.cap = None

        # Get time of initiation.
        self.fps = FPS().start()

    def capture(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
        except:
            pass

    def release(self):
        try:
            self.cap.release()
        except:
            pass

    # def __del__(self):
    #     # # self.video.release()
    #     # self.video.stop()
    #     pass

    # def cap(self):
    #     """cap recording device. """
    #     try:
    #         self.video = cv2.Videocap(0)
    #     except:  # TODO: catch a proper exception
    #         self.logger.error("Could not cap device camera!")

    # def release(self):
    #     self.video.release()

    def read(self):
        """read received raw frame, calculate fps, and perform simple preprocess. """

        # ret, image = self.video.read()
        (self.grabbed, self.frame) = self.cap.read()
        image = self.frame

        if image is not None:
            """Update FPS, and incode received frame. """
            self.fps.update()
            # TODO: add self.fps.fps() to image, if flagged raised.

            # We are using Motion JPEG, but OpenCV defaults to cap raw images,
            # so we must encode it into JPEG in order to correctly display the
            # video stream.

            # display a piece of text to the frame (so we can benchmark
            # fairly against the fast method)
            self.fps.stop()
            cv2.putText(image, "FPS (simple): {:.2f}".format(self.fps.fps()), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.frame = image.copy()

            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            self.logger.debug("in 'get_frame', video.read not success")


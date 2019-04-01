"""Camera module,

capture raw frame,
then pass it to SimpleCamera & SmartCamera.
"""

import logging
import cv2
from utils.simple_camera import SimpleCamera

# from utils.smart_camera import SmartCamera


# Create loggers.
camera_logger = logging.getLogger('camera_module')
ch = logging.StreamHandler()
# create formatter and add it to the handlers.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to loggers.
camera_logger.addHandler(ch)


class CameraModule:

    def __init__(self, logger=None):
        self.frame = None
        self.logger = logging.getLogger('camera_module') if logger is None else logger
        self.toggle_flag = True

        # CameraModule will capture a frame, then pipe it to smart/simple cameras.
        self.capture = cv2.VideoCapture(0)
        # self.capture.release

        self.simple_camera = SimpleCamera()

    #       self.smart_camera = SmartCamera()

    def toggle_camera_modes(self):
        if self.toggle_flag is True:
            self.toggle_flag = False
            self.logger.info("simple cam: ON")
            try:
                self.smart_camera.release()
            except:
                pass

            # self.simple_camera.capture()
            # self.frame = self.simple_camera.get_frame()

    def read(self):
        (self.grabbed, self.frame) = self.capture.read()
        if self.toggle_flag is True:
            self.logger.info("getting frame from simple cam")
            return self.simple_camera.read(self.frame)

#     def show(self):
#         """For debugging. """
#         cv2.imshow(" ", self.frame)
#         cv2.waitKey(0)
#
#
# if __name__ == "__main__":
#     camera_module = CameraModule()
#     camera_module.read()
#     camera_module.show()
#     camera_module.capture.release()

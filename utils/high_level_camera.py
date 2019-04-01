"""Camera module,

capture raw frame,
then pass it to SimpleCamera & SmartCamera.
"""

import logging
import time

import cv2
from utils.simple_camera import SimpleCamera

# from utils.smart_camera import SmartCamera


# Create loggers.
from utils.smart_camera import SmartCamera

camera_logger = logging.getLogger('camera_module')
ch = logging.StreamHandler()
# create formatter and add it to the handlers.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to loggers.
camera_logger.addHandler(ch)


class CameraModule:

    def __init__(self, logger=None):
        """CameraModule is initiated with toggle_flag == True,

        which means it is sets to simple camera.
        """

        self.frame = None
        self.logger = logging.getLogger('camera_module') if logger is None else logger
        self.toggle_flag = True

        # CameraModule will capture a frame, then pipe it to smart/simple cameras.
        # self.capture = cv2.VideoCapture(0)
        # self.capture.release

        self.simple_camera = SimpleCamera()
        self.smart_camera = SmartCameKra()

    def toggle_camera_modes(self):
        if self.toggle_flag is True:
            """Then user wants to change from simple to smart cameras. """

            self.toggle_flag = False
            self.logger.info("turning simple cam: OFF")
            self.logger.info("turning smart cam: ON")
            try:
                self.simple_camera.release()
                time.sleep(1)
                self.smart_camera.capture()
            except:
                pass

        elif self.toggle_flag is False:
            """Then user wants to change from simple to simple cameras. """

            self.toggle_flag = True
            self.logger.info("turning simple cam: ON")
            self.logger.info("turning smart cam: OFF")
            try:
                self.smart_camera.release()
                time.sleep(1)
                self.simple_camera.capture()
            except:
                pass

    def read(self):
        if self.toggle_flag is True:
            self.logger.info("getting frame from simple cam")
            return self.simple_camera.read()

        elif self.toggle_flag is False:
            self.logger.info("getting frame from smart cam")
            return self.smart_camera.read()


    def release(self):
        try:
            self.simple_camera.release()
        except:
            pass

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

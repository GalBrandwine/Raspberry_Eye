"""Camera module,

capture raw frame,
then pass it to SimpleCamera & SmartCamera.
"""

import logging
import time

from utils.simple_camera import SimpleCamera
from utils.smart_camera import SmartCamera

# Create loggers.
logging.basicConfig(level=logging.INFO)
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
        self.smart_flag = None
        self.simple_flag = True

        # CameraModule will capture a frame, then pipe it to smart/simple cameras.
        # self.capture = cv2.VideoCapture(0)
        # self.capture.release

        self.simple_camera = SimpleCamera()
        self.smart_camera = SmartCamera("/media/gal/DATA/Documents/projects/Raspberry_Eye/graphs/mobilenetgraph")
        self.init()

    def toggle_camera_modes(self, mode):
        """toggle hl_camera mods controlls in which modes a camera should owrk.

        params: smart_on, smart_off(simple), cat, dog.
        """

        if mode is "smart_on":
            """Then user wants to change from simple to smart cameras. 
            
            """

            self.smart_flag = True
            self.logger.info("turning simple cam: OFF")
            self.logger.info("turning smart cam: ON")
            try:
                if self.simple_flag is True:
                    self.simple_flag  = False
                    self.simple_camera.release()
                    time.sleep(1)
                    self.smart_camera.capture(None)
                else:
                    # Change mode None, meaning no object tracknig. just user neuronNet

                    self.smart_camera.object_to_track = None
            except:
                pass

        elif mode is "smart_off":
            """Then user wants to change from simple to simple cameras. """

            self.smart_flag = False
            self.simple_flag = True
            self.logger.info("turning simple cam: ON")
            self.logger.info("turning smart cam: OFF")
            try:
                self.smart_camera.release()
                time.sleep(1)
                self.simple_camera.capture()
            except:
                pass

        elif mode is "dog" and self.smart_flag is True:
            """Then user wants to change camera mode to track_a_dog. accesible only if smart is on. """

            # self.toggle_flag = True
            self.logger.info("turning smart cam: ON, in dog mode")
            self.smart_camera.object_to_track = mode

        elif mode is "cat" and self.smart_flag is True:
            """Then user wants to change camera mode to track_a_cat. accesible only if smart is on. """

            self.logger.info("turning smart cam: ON, in cat mode")
            self.smart_camera.object_to_track = mode

    def read(self):
        if self.simple_flag is True:
            # self.logger.info("getting frame from simple cam")
            return self.simple_camera.read()

        elif self.smart_flag is True:
            # self.logger.info("getting frame from smart cam")
            return self.smart_camera.read()

    def release(self):
        try:
            self.simple_camera.release()
        except:
            pass

    def init(self):
        self.simple_camera.capture()

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

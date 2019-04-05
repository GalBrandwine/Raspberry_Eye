import cv2
import numpy as np

from raspberry_eye.simple_camera import SimpleCamera


class TestSimpleCamera:
    def test_read(self):
        """Simple test to see if simple_camera read frame, than convert it properly to bytecood.

        Test passes if both images look the same.
        """

        # setup
        simple_camera = SimpleCamera()
        image_bytes = simple_camera.read()
        decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

        #run
        cv2.imshow("simple camera frame", simple_camera.frame)
        cv2.imshow("jped_incoded_ returned from simple camera", decoded)
        cv2.waitKey(0)


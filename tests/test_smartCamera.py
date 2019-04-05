import cv2
import numpy as np
from raspberry_eye.smart_camera import SmartCamera


class TestSmartCamera:

    def test_capture(self):
        """Test if smart_camera capture the video feed when it is available. """
        self.fail()

    def test_capture_when_captured(self):
        """Test if smart_camera capture the video feed when it already captured. """
        self.fail()

    def test_release(self):
        """Test if smart_camera releases video feed correctly. """
        self.fail()

    def test_read(self):
        """Simple test to see if smart_camera read frame, than convert it properly to bytecood.

        Test passes if both images look the same.
        """
        # setup
        smart_camera = SmartCamera("/media/gal/DATA/Documents/projects/Raspberry_Eye/graphs/mobilenetgraph")
        smart_camera.capture()
        image_bytes = smart_camera.read()
        decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

        # run
        cv2.imshow("smart camera frame", smart_camera.frame)
        cv2.imshow("jped_incoded_ returned from smart camera", decoded)
        cv2.waitKey(0)

    def test_preprocess_image(self):
        self.fail()

    def test_predict(self):
        self.fail()

    def test_ncs_init(self):
        self.fail()

    def test_ncs_predict(self):
        self.fail()

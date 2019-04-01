import cv2
import numpy as np
from utils.high_level_camera import CameraModule


class TestCameraModule:

    def test_hl_camera_shows_simple_camera(self):
        """Test will pass if simple camera working correctly, within hl_camera. """

        # setup
        camera_module = CameraModule()

        # run
        while True:
            # grab the frame from the camera_module (set to simple)
            # the frame is bytecoded, so decode it back to numpy
            image_bytes = camera_module.read()
            decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

            # run
            cv2.imshow("camera_module in simple camera mode", decoded)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # teardown
        camera_module.release()

    def test_toggle_camera_modes(self):
        """This test will pass if the camera's switching correctly. """

        # setup
        camera_module = CameraModule()
        # feeder_mock = VideoFeederMock()

        # run
        while True:
            # grab the frame from the camera_module (set to simple)
            # the frame is bytecoded, so decode it back to numpy
            image_bytes = camera_module.read()
            decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

            # run
            cv2.imshow("camera_module in simple camera mode", decoded)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                camera_module.toggle_camera_modes()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # teardown
        camera_module.release()

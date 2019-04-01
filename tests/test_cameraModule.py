import cv2

from utils.high_level_camera import CameraModule


class VideoFeederMock:

    def __init__(self):
        self.cameraMock = None

    def capture(self, camera=None):
        if self.cameraMock is None:
            self.cameraMock = cv2.VideoCapture(0)
        else:
            self.cameraMock = camera

        # loop over frames from the video file stream
        while True:
            # grab the frame from the threaded video file stream
            (grabbed, frame) = self.cameraMock.read()

            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break

            # resize the frame and convert it to grayscale (while still
            # retaining 3 channels)
            # frame = imutils.resize(frame, width=450)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame = np.dstack([frame, frame, frame])

            # display a piece of text to the frame (so we can benchmark
            # fairly against the fast method)
            # cv2.putText(frame, "Slow Method", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # show the frame and update the FPS counter
            cv2.imshow("Frame_Mock", frame)
            cv2.waitKey(1)

    def release(self):
        pass


class TestCameraModule:

    def test_toggle_camera_modes(self):
        """This test will pass if the camera's switching correctly. """

        #setup
        camera_module = CameraModule()
        # feeder_mock = VideoFeederMock()

        #run
        while True:
            # grab the frame from the threaded video file stream
            frame = camera_module.read()
            camera_module.show()

        self.fail()

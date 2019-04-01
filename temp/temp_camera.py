import cv2

"""
There are a few issues with your code:

    There is no need to initialize the camera in the __init__(self) function. Why? You are already calling it in get_frame(self).
    
    In function get_frame(self), at the end it returns self.frames.read() at the end. 
    You are supposed to return the image captured by self.cap.read(). This resulted in AttributeError.
    
    I also added Camera().release_camera() to turn off the webcam once the execution is over.

Here is the restructured code (I did not use imutils, I just used cv2.resize()):
"""


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Prepare the camera...
        print("Camera warming up ...")

    def get_frame(self):
        s, img = self.cap.read()
        if s:  # frame captures without errors...

            pass

        return img

    def release_camera(self):
        self.cap.release()


def main():
    cam1 = Camera()
    while True:
        frame = cam1.get_frame()
        frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    Camera().release_camera()
    return ()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()

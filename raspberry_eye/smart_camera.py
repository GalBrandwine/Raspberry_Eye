"""A smart webcam,

 get video feed from camera,
 feedforward captuerd frame on NCS pretrained graph,
 compute FPS,
 return the captured frame, with detected object, inframe_self.logger.infoed FPS, and header: "smart camera".
 """

# USAGE
# python ncs_realtime_objectdetection.py --graph graphs/mobilenetgraph --display 1
# python ncs_realtime_objectdetection.py --graph graphs/mobilenetgraph --confidence 0.5 --display 1

try:
    from mvnc import mvncapi as mvnc

except ImportError as err:
    """For none mvnc environments. """


    class mvnc:
        """A mock nvmc."""

        devices = [0, 1]

        @classmethod
        def EnumerateDevices(cls):
            return cls.devices

        @classmethod
        def Device(cls, param):
            class device:
                def OpenDevice(self):
                    return 1

                def AllocateGraph(self, graph_in_memory):
                    graph = 1
                    return graph

            return device()

import signal
import sys
import time
# import necessary packages
from multiprocessing import Manager
from multiprocessing import Process



from raspberry_eye.pan_tilt import pan_tilt as PanTilt
from raspberry_eye.pid_controller.objcenter import ObjCenter
from raspberry_eye.pid_controller.pid import PID

import os
from _pytest import logging
from imutils.video import FPS
from imutils.video import VideoStream
import numpy as np
import logging
import cv2

# Init logger
camera_logger = logging.getLogger('Smart_camera')
ch = logging.StreamHandler()
# create formatter and add it to the handlers.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to loggers.
camera_logger.addHandler(ch)

    
# frame dimensions should be sqaure
PREPROCESS_DIMS = (300, 300)
DISPLAY_DIMS = (900, 900)

# calculate the multiplier needed to scale the bounding boxes
DISP_MULTIPLIER = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]

# define the range for the motors
servoRange = (-90, 90)


# function to handle keyboard interrupt
def signal_handler(sig, frame):
    # print a status message
    print("[INFO] You pressed `ctrl + c`! Exiting...")

    # disable the servos
    PanTilt.servo_enable(1, False)
    PanTilt.servo_enable(2, False)

    # exit
    sys.exit()


def obj_center(args, objX, objY, centerX, centerY):
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    # start the video stream and wait for the camera to warm up
    vs = VideoStream(usePiCamera=False).start()
    time.sleep(2.0)

    # initialize the object center finder
    # obj = ObjCenter(args["cascade"])
    obj = ObjCenter("/home/pi/Gimbal_Pi/pan_tilt_tracking/haar.xml")
        
    img = vs.read()
    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # loop indefinitely
    while True:
        # grab the frame from the threaded video stream and flip it
        # vertically (since our camera was upside down)
        frame = vs.read()
        # frame = cv2.flip(frame, 0)

        # Resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        # Overcome mirror effect
        frame = cv2.flip(frame, 1)

        # calculate the center of the frame as this is where we will
        # try to keep the object
        (H, W) = frame.shape[:2]
        centerX.value = W // 2
        centerY.value = H // 2
        cv2.circle(frame,(centerX.value,centerY.value), 5, (0,0,255), -1)
        
        ## find the object's location
        objectLoc = obj.update(frame, (centerX.value, centerY.value))
        
        ((objX.value, objY.value), rect) = objectLoc
        cv2.circle(frame,(objX.value, objY.value), 5, (0,255,0), -1)

        ## extract the bounding box and draw it
        if rect is not None:
            (ptA, ptB) = rect
            cv2.rectangle(frame, ptA, ptB, (0, 255, 0),
                          2)
                     
        # display the frame to the screen
        cv2.imshow("Pan-Tilt Face Tracking", frame)
        cv2.waitKey(1)

def pid_process(output, p, i, d, objCoord, centerCoord):
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    # create a PID and initialize it
    p = PID(p.value, i.value, d.value)
    p.initialize()

    # loop indefinitely
    while True:
        # calculate the error
        error = centerCoord.value - objCoord.value
        #print(centerCoord.value,objCoord.value,error)
        # update the value
        output.value = p.update(error)


def in_range(val, start, end):
    # determine the input vale is in the supplied range
    return (val >= start and val <= end)


def go(pan, tlt):
    # signal trap to handle keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)
    
    # loop indefinitely
    while True:
        # the pan and tilt angles are reversed
        panAngle = 1 * pan.value
        tltAngle = 1 * tlt.value

        # if the pan angle is within the range, pan
        if in_range(panAngle, servoRange[0], servoRange[1]):
            PanTilt.pan(panAngle)

        # if the tilt angle is within the range, tilt
        if in_range(tltAngle, servoRange[0], servoRange[1]):
            PanTilt.tilt(tltAngle)


            
            
# noinspection PyMethodMayBeStatic
class SmartCamera:
    def __init__(self, graph_path, camera_index, logger=None):
        """Using OpenCV to capture from device 0.

        If you have trouble capturing
        from a Web-cam, comment the line below out and use a video file
        instead.
        """
        self.graph_path = graph_path
        self.camera_index = camera_index
        self.graph = None
        self.device = None
        self.graph_in_memory = None
        self.object_to_track = None

        self.logger = logging.getLogger('Smart_camera') if logger is None else logger
        self.grabbed = None
        self.frame = None

        self.cap = None

        # Get time of initiation.
        self.fps = FPS().start()
        # self.__ncs_init(self.graph_path)

    def capture(self, object_to_track=None):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            # Get time of initiation.
            self.fps = FPS().start()
            self.object_to_track = object_to_track
            self.__ncs_init()

        except Exception as exception:
            self.logger.error("In smart_camera.capture: {}".format(exception))
        except AttributeError as err:
            self.logger.error(err)

    def release(self):
        try:
            # clean up camera capture, graph and device
            self.cap.release()
            self.graph.DeallocateGraph()
            self.device.CloseDevice()
            # Get time of initiation.
            self.fps = FPS().stop()

        except Exception as exception:
            self.logger.error("In smart_camera.release: {}".format(exception))

        except AttributeError as err:
            self.logger.error(err)

    def read(self):
        """Read received raw frame, calculate fps, and perform smart preprocess. """

        (self.grabbed, self.frame) = self.cap.read()
        image = self.frame

        if image is not None:
            """Update FPS, and incode received frame. """
            self.fps.update()

            # Start the NCS processing pipeline.
            image_with_predictions = self.__ncs_predict(image)
            if image_with_predictions is None:
                height, width = image.shape[:2]
                image_with_predictions = np.zeros((height, width, 3), np.uint8)

                cv2.putText(image_with_predictions,
                            "{}".format("Error in smart_camera.read: image_with_predictions=None"), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # todo: If self.object_to_track is not NONE, call gimbal.auto_tracker(self.object_to_track).
                # todo: DEVELOP gimbal module.
                if self.object_to_track is not None:
                    self.logger.info("smart camera tracking: {}".format(self.object_to_track))
                    
                
                # We are using Motion JPEG, but OpenCV defaults to capture raw images,
                # so we must encode it into JPEG in order to correctly display the
                # video stream.

                # Display a piece of text to the frame (so we can benchmark)
                self.fps.stop()
                cv2.putText(image_with_predictions, "FPS (smart): {:.2f}".format(self.fps.fps()), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # self.frame = image_with_predictions.copy()
            ret, jpeg = cv2.imencode('.jpg', image_with_predictions)
            return jpeg.tobytes()
        else:
            self.logger.debug("in 'get_frame', video.read not success")
            return None

    # ------------- Preprocessing functions (private) -------------
    def __preprocess_image(self, input_image):
        # preprocess the image
        preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
        preprocessed = preprocessed - 127.5
        preprocessed = preprocessed * 0.007843
        preprocessed = preprocessed.astype(np.float16)

        # return the image to the calling function
        return preprocessed

    def __predict(self, image, graph):

        predictions = []
        # preprocess the image
        image = self.__preprocess_image(image)

        # send the image to the NCS and run a forward pass to grab the
        # network predictions
        try:
            graph.LoadTensor(image, None)
            (output, _) = graph.GetResult()

        # Rolling the exception upward.
        # except Exception as INVALID_PARAMETERS:
        #     self.logger.error(INVALID_PARAMETERS)
        #     return None

        except TypeError as err:
            self.logger.error(err)
            return

        # grab the number of valid object predictions from the output,
        # then initialize the list of predictions
        num_valid_boxes = output[0]

        # loop over results
        for box_index in range(num_valid_boxes):
            # calculate the base index into our array so we can extract
            # bounding box information
            base_index = 7 + box_index * 7

            # boxes with non-finite (inf, nan, etc) numbers must be ignored
            if (not np.isfinite(output[base_index]) or
                    not np.isfinite(output[base_index + 1]) or
                    not np.isfinite(output[base_index + 2]) or
                    not np.isfinite(output[base_index + 3]) or
                    not np.isfinite(output[base_index + 4]) or
                    not np.isfinite(output[base_index + 5]) or
                    not np.isfinite(output[base_index + 6])):
                continue

            # extract the image width and height and clip the boxes to the
            # image size in case network returns boxes outside of the image
            # boundaries
            (h, w) = image.shape[:2]
            x1 = max(0, int(output[base_index + 3] * w))
            y1 = max(0, int(output[base_index + 4] * h))
            x2 = min(w, int(output[base_index + 5] * w))
            y2 = min(h, int(output[base_index + 6] * h))

            # grab the prediction class label, confidence (i.e., probability),
            # and bounding box (x, y)-coordinates
            pred_class = int(output[base_index + 1])
            pred_conf = output[base_index + 2]
            pred_boxpts = ((x1, y1), (x2, y2))

            # create prediciton tuple and append the prediction to the
            # predictions list
            prediction = (pred_class, pred_conf, pred_boxpts)
            predictions.append(prediction)

        # return the list of predictions to the calling function
        return predictions

    def __ncs_init(self, graph_path=None, confidence=None, display=None):
        # grab a list of all NCS devices plugged in to USB
        self.logger.info("[INFO] finding NCS devices...")
        self.devices = mvnc.EnumerateDevices()

        # if no devices found, exit the script
        if len(self.devices) == 0:
            self.logger.info("[INFO] No devices found. Please plug in a NCS")
            quit()

        # use the first device since this is a simple test script
        # (you'll want to modify this is using multiple NCS devices)
        self.logger.info("[INFO] found {} devices. device0 will be used. "
                         "opening device0...".format(len(self.devices)))
        self.device = mvnc.Device(self.devices[0])
        self.device.OpenDevice()

        if self.graph_in_memory is None:
            # open the CNN graph file
            self.logger.info("[INFO] loading the graph file into RPi memory...")
            with open(self.graph_path, mode="rb") as f:
                self.graph_in_memory = f.read()

        # load the graph into the NCS
        self.logger.info("[INFO] allocating the graph on the NCS...")
        self.graph = self.device.AllocateGraph(self.graph_in_memory)

    def __ncs_predict(self, frame=None):
        try:
            # grab the frame from the threaded video stream
            # make a copy of the frame and resize it for display/video purposes
            frame = frame
            image_for_result = frame.copy()
            image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)

            # use the NCS to acquire predictions
            try:
                predictions = self.__predict(frame, self.graph)
            except TypeError as err:
                self.logger.error("in __ncs_predict: {}".format(err))
                # there's a bug: mvns.INVALID_PARAMETERS
                # if en exception thrown, return an unpredicted image.
                cv2.putText(image_for_result, "             {}".format(err), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                return image_for_result

            # loop over our predictions
            for (i, pred) in enumerate(predictions):
                # extract prediction data for readability
                (pred_class, pred_conf, pred_boxpts) = pred

                # filter out weak detections by ensuring the `confidence`
                # is greater than the minimum confidence
                if pred_conf > 0.5:
                    # self.logger.info prediction to terminal
                    # self.logger.info("[INFO] prediction #{}: class={}, confidence={}, "
                    #                  "boxpoints={}".format(i, CLASSES[pred_class], pred_conf,
                    #                                        pred_boxpts))

                    # build a label consisting of the predicted class and
                    # associated probability
                    label = "{}: {:.2f}%".format(CLASSES[pred_class],
                                                 pred_conf * 100)

                    # extract information from the prediction boxpoints
                    (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
                    ptA = (ptA[0] * DISP_MULTIPLIER, ptA[1] * DISP_MULTIPLIER)
                    ptB = (ptB[0] * DISP_MULTIPLIER, ptB[1] * DISP_MULTIPLIER)
                    (startX, startY) = (ptA[0], ptA[1])
                    y = startY - 15 if startY - 15 > 15 else startY + 15

                    # display the rectangle and label text
                    cv2.rectangle(image_for_result, ptA, ptB,
                                  COLORS[pred_class], 2)
                    cv2.putText(image_for_result, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 3)

            # update the FPS counter
            self.fps.update()
            return image_for_result
        # if there's a problem reading a frame, break gracefully
        except AttributeError as err:
            self.logger.error(err)
            
# check to see if this is the main body of execution
if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-c", "--cascade", type=str, required=False,
                    #help="path to input Haar cascade for face detection")
    #args = vars(ap.parse_args())
    print("srarted")
    # start a manager for managing process-safe variables
    with Manager() as manager:
        # enable the servos
        PanTilt.servo_enable(1, True)
        PanTilt.servo_enable(2, True)

        # set integer values for the object center (x, y)-coordinates
        centerX = manager.Value("i", 0)
        centerY = manager.Value("i", 0)

        # set integer values for the object's (x, y)-coordinates
        objX = manager.Value("i", 0)
        objY = manager.Value("i", 0)

        # pan and tilt values will be managed by independed PIDs
        pan = manager.Value("i", 0)
        tlt = manager.Value("i", 0)

        # set PID values for panning
        panP = manager.Value("f", 0.09)
        panI = manager.Value("f", 0.08)
        panD = manager.Value("f", 0.002)

        # set PID values for tilting
        tiltP = manager.Value("f", 0.11)
        tiltI = manager.Value("f", 0.10)
        tiltD = manager.Value("f", 0.002)

        # we have 4 independent processes
        # 1. objectCenter  - finds/localizes the object
        # 2. panning       - PID control loop determines panning angle
        # 3. tilting       - PID control loop determines tilting angle
        # 4. setServos     - drives the servos to proper angles based
        #                    on PID feedback to keep object in center
        args = None
        processObjectCenter = Process(target=obj_center,
                                      args=(args, objX, objY, centerX, centerY))
        processPanning = Process(target=pid_process,
                                 args=(pan, panP, panI, panD, objX, centerX))
        processTilting = Process(target=pid_process,
                                 args=(tlt, tiltP, tiltI, tiltD, objY, centerY))
        processSetServos = Process(target=go, args=(pan, tlt))

        # start all 4 processes
        processObjectCenter.start()
        processPanning.start()
        processTilting.start()
        processSetServos.start()

        # join all 4 processes
        processObjectCenter.join()
        processPanning.join()
        processTilting.join()
        processSetServos.join()

        # disable the servos
        PanTilt.servo_enable(1, False)
        PanTilt.servo_enable(2, False)


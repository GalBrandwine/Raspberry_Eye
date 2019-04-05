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

import os
from _pytest import logging
from imutils.video import FPS
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

# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ("background", "aeroplane", "bicycle", "bird",
           "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# frame dimensions should be sqaure
PREPROCESS_DIMS = (300, 300)
DISPLAY_DIMS = (900, 900)

# calculate the multiplier needed to scale the bounding boxes
DISP_MULTIPLIER = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]


# noinspection PyMethodMayBeStatic
class SmartCamera:
    def __init__(self, graph_path, logger=None):
        """Using OpenCV to capture from device 0.

        If you have trouble capturing
        from a Web-cam, comment the line below out and use a video file
        instead.
        """
        self.graph_path = graph_path
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
        self.__ncs_init(self.graph_path)

    def capture(self, object_to_track=None):
        try:
            self.cap = cv2.VideoCapture(0)
            self.object_to_track = object_to_track
        except AttributeError as err:
            self.logger.error(err)

    def release(self):
        try:
            self.cap.release()
            # clean up the graph and device
            self.graph.DeallocateGraph()
            self.device.CloseDevice()

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

            # todo: If self.object_to_track is not NONE, call gimbal.auto_tracker(self.object_to_track).
            # todo: DEVELOP gimbal module.
            # self.logger.info("smart camera tracking: {}".format(self.object_to_track))
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
            pass

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

        # open the CNN graph file
        self.logger.info("[INFO] loading the graph file into RPi memory...")
        with open(graph_path, mode="rb") as f:
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
            except:
                # there's a bug: mvns.INVALID_PARAMETERS
                # if en exception thrown, return an unpredicted image.
                cv2.putText(image_for_result, "FPS (smart): Error in predictions".format(self.fps.fps()), (10, 30),
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

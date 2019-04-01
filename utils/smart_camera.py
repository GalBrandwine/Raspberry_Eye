"""A smart webcam,

 get video feed from camera,
 feedforward captuerd frame on NCS pretrained graph,
 compute FPS,
 return the captured frame, with detected object, inframe_self.logger.infoed FPS, and header: "smart camera".
 """

# USAGE
# python ncs_realtime_objectdetection.py --graph graphs/mobilenetgraph --display 1
# python ncs_realtime_objectdetection.py --graph graphs/mobilenetgraph --confidence 0.5 --display 1

# import the necessary packages
from _pytest import logging
from mvnc import mvncapi as mvnc
from imutils.video import FPS
import numpy as np
import time
import cv2

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


class SmartCamera:
    def __init__(self, logger=None):
        """Using OpenCV to capture from device 0.

        If you have trouble capturing
        from a Web-cam, comment the line below out and use a video file
        instead.
        """
        self.graph = None
        self.graph_in_memory = None
        self.devices = mvnc.EnumerateDevices()
        self.predictions = []
        self.logger = logging.getLogger('camera_handler') if logger is None else logger
        self.grabbed = None
        self.frame = None

        # Threaded attempt for frame streaming, maybe not necessary.
        # self.video = WebcamVideoStream(src=0).start()
        self.capture = cv2.VideoCapture(0)
        # Get time of initiation.
        self.fps = FPS().start()
        self.ncs_init()

    def capture(self):
        try:
            self.capture = cv2.VideoCapture(0)
        except AttributeError as err:
            self.logger.error(err)

    def release(self):
        try:
            self.capture().release()
            # clean up the graph and device
            self.graph.DeallocateGraph()
            self.device.CloseDevice()

        except AttributeError as err:
            self.logger.error(err)

    def read(self):
        """read received raw frame, calculate fps, and perform smart preprocess. """

        (self.grabbed, self.frame) = self.capture.read()
        image = self.frame

        if image is not None:
            """Update FPS, and incode received frame. """
            self.fps.update()

            # ncs_predict
            self.ncs_predict()

            # We are using Motion JPEG, but OpenCV defaults to capture raw images,
            # so we must encode it into JPEG in order to correctly display the
            # video stream.

            # display a piece of text to the frame (so we can benchmark
            # fairly against the fast method)
            self.fps.stop()
            cv2.putText(image, "FPS: {:.2f}".format(self.fps.fps()), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.frame = image.copy()

            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            self.logger.debug("in 'get_frame', video.read not success")

    # ------------- Preprocessing functions -------------
    def preprocess_image(self, input_image):
        # preprocess the image
        preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
        preprocessed = preprocessed - 127.5
        preprocessed = preprocessed * 0.007843
        preprocessed = preprocessed.astype(np.float16)

        # return the image to the calling function
        return preprocessed

    def predict(self, image, graph):
        # preprocess the image
        image = self.preprocess_image(image)

        # send the image to the NCS and run a forward pass to grab the
        # network predictions
        graph.LoadTensor(image, None)
        (output, _) = graph.GetResult()

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
            self.predictions.append(prediction)

        # return the list of predictions to the calling function
        return self.predictions

    def ncs_init(self, graph_path=None, confidence=None, display=None):
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
        device = mvnc.Device(self.devices[0])
        device.OpenDevice()

        # open the CNN graph file
        self.logger.info("[INFO] loading the graph file into RPi memory...")
        with open(graph_path, mode="rb") as f:
            self.graph_in_memory = f.read()

        # load the graph into the NCS
        self.logger.info("[INFO] allocating the graph on the NCS...")
        self.graph = device.AllocateGraph(self.graph_in_memory)

    def ncs_predict(self):
        try:
            # grab the frame from the threaded video stream
            # make a copy of the frame and resize it for display/video purposes
            frame = vs.read()
            image_for_result = frame.copy()
            image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)

            # use the NCS to acquire predictions
            predictions = predict(frame, graph)

            # loop over our predictions
            for (i, pred) in enumerate(predictions):
                # extract prediction data for readability
                (pred_class, pred_conf, pred_boxpts) = pred

                # filter out weak detections by ensuring the `confidence`
                # is greater than the minimum confidence
                if pred_conf > args["confidence"]:
                    # self.logger.info prediction to terminal
                    self.logger.info("[INFO] Prediction #{}: class={}, confidence={}, "
                                     "boxpoints={}".format(i, CLASSES[pred_class], pred_conf,
                                                           pred_boxpts))

                    # check if we should show the prediction data
                    # on the frame
                    if args["display"] > 0:
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

            # check if we should display the frame on the screen
            # with prediction data (you can achieve faster FPS if you
            # do not output to the screen)
            # if args["display"] > 0:
            #     # display the frame to the screen
            #     cv2.imshow("Output", image_for_result)
            #     key = cv2.waitKey(1) & 0xFF
            #
            #     # if the `q` key was pressed, break from the loop
            #     if key == ord("q"):
            #         break

            # update the FPS counter
            fps.update()

        # if "ctrl+c" is pressed in the terminal, break from the loop
        except KeyboardInterrupt as err:
            self.logger.error(err)

        # if there's a problem reading a frame, break gracefully
        except AttributeError as err:
            self.logger.error(err)


# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-g", "--graph", required=True,
#                 help="path to input graph file")
# ap.add_argument("-c", "--confidence", default=.5,
#                 help="confidence threshold")
# ap.add_argument("-d", "--display", type=int, default=0,
#                 help="switch to display image on screen")
# args = vars(ap.parse_args())

# # grab a list of all NCS devices plugged in to USB
# self.logger.info("[INFO] finding NCS devices...")
# devices = mvnc.EnumerateDevices()
# 
# # if no devices found, exit the script
# if len(devices) == 0:
#     self.logger.info("[INFO] No devices found. Please plug in a NCS")
#     quit()
# 
# # use the first device since this is a simple test script
# # (you'll want to modify this is using multiple NCS devices)
# self.logger.info("[INFO] found {} devices. device0 will be used. "
#       "opening device0...".format(len(devices)))
# device = mvnc.Device(devices[0])
# device.OpenDevice()

# open the CNN graph file
# self.logger.info("[INFO] loading the graph file into RPi memory...")
# with open(args["graph"], mode="rb") as f:
#     graph_in_memory = f.read()
#
# # load the graph into the NCS
# self.logger.info("[INFO] allocating the graph on the NCS...")
# graph = device.AllocateGraph(graph_in_memory)

# open a pointer to the video stream thread and allow the buffer to
# start to fill, then start the FPS counter
# self.logger.info("[INFO] starting the video stream and FPS counter...")
# # vs = VideoStream(usePiCamera=False).start()
# vs = WebcamVideoStream(src=0).start()
# time.sleep(1)
# fps = FPS().start()
#
# # loop over frames from the video file stream
# while True:
#     try:
#         # grab the frame from the threaded video stream
#         # make a copy of the frame and resize it for display/video purposes
#         frame = vs.read()
#         image_for_result = frame.copy()
#         image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)
#
#         # use the NCS to acquire predictions
#         predictions = predict(frame, graph)
#
#         # loop over our predictions
#         for (i, pred) in enumerate(predictions):
#             # extract prediction data for readability
#             (pred_class, pred_conf, pred_boxpts) = pred
#
#             # filter out weak detections by ensuring the `confidence`
#             # is greater than the minimum confidence
#             if pred_conf > args["confidence"]:
#                 # self.logger.info prediction to terminal
#                 self.logger.info("[INFO] Prediction #{}: class={}, confidence={}, "
#                                  "boxpoints={}".format(i, CLASSES[pred_class], pred_conf,
#                                                        pred_boxpts))
#
#                 # check if we should show the prediction data
#                 # on the frame
#                 if args["display"] > 0:
#                     # build a label consisting of the predicted class and
#                     # associated probability
#                     label = "{}: {:.2f}%".format(CLASSES[pred_class],
#                                                  pred_conf * 100)
#
#                     # extract information from the prediction boxpoints
#                     (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
#                     ptA = (ptA[0] * DISP_MULTIPLIER, ptA[1] * DISP_MULTIPLIER)
#                     ptB = (ptB[0] * DISP_MULTIPLIER, ptB[1] * DISP_MULTIPLIER)
#                     (startX, startY) = (ptA[0], ptA[1])
#                     y = startY - 15 if startY - 15 > 15 else startY + 15
#
#                     # display the rectangle and label text
#                     cv2.rectangle(image_for_result, ptA, ptB,
#                                   COLORS[pred_class], 2)
#                     cv2.putText(image_for_result, label, (startX, y),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 3)
#
#         # check if we should display the frame on the screen
#         # with prediction data (you can achieve faster FPS if you
#         # do not output to the screen)
#         if args["display"] > 0:
#             # display the frame to the screen
#             cv2.imshow("Output", image_for_result)
#             key = cv2.waitKey(1) & 0xFF
#
#             # if the `q` key was pressed, break from the loop
#             if key == ord("q"):
#                 break
#
#         # update the FPS counter
#         fps.update()
#
#     # if "ctrl+c" is pressed in the terminal, break from the loop
#     except KeyboardInterrupt:
#         break
#
#     # if there's a problem reading a frame, break gracefully
#     except AttributeError:
#         break
#
# # stop the FPS counter timer
# fps.stop()
#
# # destroy all windows if we are displaying them
# if args["display"] > 0:
#     cv2.destroyAllWindows()

# stop the video stream
vs.stop()

# clean up the graph and device
graph.DeallocateGraph()
device.CloseDevice()

# display FPS information
self.logger.info("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
self.logger.info("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

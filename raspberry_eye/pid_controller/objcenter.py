# import necessary packages
import imutils
import cv2
import numpy as np
import logging

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

# Init NCS
# grab a list of all NCS devices plugged in to USB
camera_logger.info("[INFO] finding NCS devices...")
devices = mvnc.EnumerateDevices()
# if no devices found, exit the script
if len(devices) == 0:
    camera_logger.info("[INFO] No devices found. Please plug in a NCS")
    quit()

# use the first device since this is a simple test script
# (you'll want to modify this is using multiple NCS devices)
camera_logger.info("[INFO] found {} devices. device0 will be used. "
                 "opening device0...".format(len(devices)))
device = mvnc.Device(devices[0])
device.OpenDevice()


# open the CNN graph file
graph_path = "/home/pi/workspace/Raspberry_Eye/graphs/mobilenetgraph"
camera_logger.info("[INFO] loading the graph file into RPi memory...")
graph_in_memory = None
with open(graph_path, mode="rb") as f:
    graph_in_memory = f.read()

# load the graph into the NCS
camera_logger.info("[INFO] allocating the graph on the NCS...")
graph = device.AllocateGraph(graph_in_memory)


class ObjCenter:
	def __init__(self, haarPath):
		# load OpenCV's Haar cascade face detector
		#self.detector = cv2.CascadeClassifier(haarPath)
		self.detector = None
		

	def update(self, frame, frameCenter):
		# convert the frame to grayscale
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect all faces in the input frame
		#rects = self.detector.detectMultiScale(gray, scaleFactor=1.05,
			#minNeighbors=9, minSize=(30, 30),
			#flags=cv2.CASCADE_SCALE_IMAGE)
		obj_found = False	
		# use the NCS to acquire predictions
		try:
			predictions = self.predict(frame, graph)
			print(predictions)
		except TypeError as err:
			self.logger.error("in __ncs_predict: {}".format(err))
			print("in __ncs_predict: {}".format(err))
			# there's a bug: mvns.INVALID_PARAMETERS
			# if en exception thrown, return an unpredicted image.
			cv2.putText(image_for_result, "             {}".format(err), (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

			return image_for_result

		# loop over our predictions
		for (i, pred) in enumerate(predictions):
			# extract prediction data for readability
			(pred_class, pred_conf, pred_boxpts) = pred
			
			# Catch the rect of desired object to track.
			
				
			# filter out weak detections by ensuring the `confidence`
			# is greater than the minimum confidence
			if pred_conf > 0.5:
				if CLASSES[pred_class] is "person":
					obj_found = True
					(ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
					ptA = (ptA[0] * DISP_MULTIPLIER, ptA[1] * DISP_MULTIPLIER)
					ptB = (ptB[0] * DISP_MULTIPLIER, ptB[1] * DISP_MULTIPLIER)
					rect = (ptA, ptB)
					
					(startX, startY) = (ptA[0], ptA[1])
					(endX, endY) = (ptB[0], ptB[1])
					width = endX - startX
					height = endY - startY
					faceX = int((startX + width/2))
					faceY = int((startY + height/2))
					print ("person middle: {}".format((faceX,faceY)))
					cv2.circle(frame,(faceX,faceY), 50, (255,255,0), 5)
				
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
				cv2.rectangle(frame, ptA, ptB,
							  COLORS[pred_class], 2)
				cv2.putText(frame, label, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 3)
		
		#check to see if a face was found
		if obj_found is True:
			# extract the bounding box coordinates of the face and
			# use the coordinates to determine the center of the
			# face
			#cv2.circle(frame,(int((x + w/2)), int((y + h/2))), 5, (255,255,0), -1)
		 
			return ((faceX, faceY), rect)

		# otherwise no faces were found, so return the center of the
		# frame
		return (frameCenter, None)
	
        
	def predict(self, image, graph):

		predictions = []
		# preprocess the image
		image = self.preprocess_image(image)

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
			print(err)
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

	# ------------- Preprocessing functions -------------
	def preprocess_image(self, input_image):
		# preprocess the image
		preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
		preprocessed = preprocessed - 127.5
		preprocessed = preprocessed * 0.007843
		preprocessed = preprocessed.astype(np.float16)

		# return the image to the calling function
		return preprocessed

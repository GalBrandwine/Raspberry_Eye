# A file for classification and benchmarking
# https://www.pyimagesearch.com/2018/02/12/getting-started-with-the-intel-movidius-neural-compute-stick/

# import the necessary packages
from mvnc import mvncapi as mvnc
import numpy as np
import argparse
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-g", "--graph", required=True,
                help="path to graph file")
ap.add_argument("-d", "--dim", type=int, required=True,
                help="dimension of input to network")
ap.add_argument("-l", "--labels", required=True,
                help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

# load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# load the input image from disk, make a copy, resize it, and convert to float32
image_orig = cv2.imread(args["image"])
image = image_orig.copy()
image = cv2.resize(image, (args["dim"], args["dim"]))
image = image.astype(np.float32)

# load the mean file and normalize
ilsvrc_mean = np.load("ilsvrc_2012_mean.npy").mean(1).mean(1)
image[:, :, 0] = (image[:, :, 0] - ilsvrc_mean[0])
image[:, :, 1] = (image[:, :, 1] - ilsvrc_mean[1])
image[:, :, 2] = (image[:, :, 2] - ilsvrc_mean[2])

# grab a list of all NCS devices plugged in to USB
print("[INFO] finding NCS devices...")
devices = mvnc.EnumerateDevices()

# if no devices found, exit the script
if len(devices) == 0:
    print("[INFO] No devices found. Please plug in a NCS")
    quit()

# use the first device since this is a simple test script
print("[INFO] found {} devices. device0 will be used. "
      "opening device0...".format(len(devices)))
device = mvnc.Device(devices[0])
device.OpenDevice()

# open the CNN graph file
print("[INFO] loading the graph file into RPi memory...")
with open(args["graph"], mode="rb") as f:
    graph_in_memory = f.read()

# load the graph into the NCS
print("[INFO] allocating the graph on the NCS...")
graph = device.AllocateGraph(graph_in_memory)

# set the image as input to the network and perform a forward-pass to
# obtain our output classification
start = time.time()
graph.LoadTensor(image.astype(np.float16), "user object")
(preds, userobj) = graph.GetResult()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end - start))

# clean up the graph and device
graph.DeallocateGraph()
device.CloseDevice()

# sort the indexes of the probabilities in descending order (higher
# probabilitiy first) and grab the top-5 predictions
preds = preds.reshape((1, len(classes)))
idxs = np.argsort(preds[0])[::-1][:5]

# loop over the top-5 predictions and display them
for (i, idx) in enumerate(idxs):
    # draw the top prediction on the input image
    if i == 0:
        text = "Label: {}, {:.2f}%".format(classes[idx],
                                           preds[0][idx] * 100)
        cv2.putText(image_orig, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    # display the predicted label + associated probability to the
    # console
    print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
                                                            classes[idx], preds[0][idx]))

# display the output image
cv2.imshow("Image", image_orig)
cv2.waitKey(0)

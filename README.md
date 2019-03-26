# Raspberry Eye is (smart ip camera) platform for deep learning models implementations 
By acomplishing an IoT OOP drived smart ip camera, i can now design new networks, train them, and compile them to 'graph's for working with tha Movidius ncs
This project dedicated to Tom & Mufasa, my pets :)


## Intro:
After learning Deep Learning from scratch (DL4CV amazing book), and experiencing with Image Processing projects (image_tracking, helloPy) , i’ve decided it is time to combine them all together.


## The goal:
build Smart IP camera that will recognize my pets, that is gimleised and controllable via a web interface (simple)


## Project components:
* A trained Neural Network model.
* A raspberry pi 3 B+.
* Movidius NCS.
* 3 servos.
* USB camera.


## Project Requirement specifications:
* git managed
* preparing unitests and test suites.
* Raspberry_Eye web interface available from all the NET
* At least 4 fps of video stream when smart detection turned ON.


# Project building blocks (top bottom approach):
### preparing the raspberry pi & Movidius workaround:
  * Working with movidius required 2 machine's:
    * For training, and converting trained models to so called 'graph' model. Which the movidius ncs can work with.
    * For predictions (e.g raspberry pi), this machine can only be used for predictions.
      I recommend to do the following tutorials by this order:

      1.    A tutorial for getting started with movidius.
            (Without setting development machine): 
            Installing dependencies, downloading trained 'graph', and running the ncs straight away on a raspberry.

            https://www.pyimagesearch.com/2018/02/12/getting-started-with-the-intel-movidius-neural-compute-stick/



      2.    A tutorial for setting the development machine for NCS,
            And converting trained caffe model to 'graph', then running it on the predictions machine.

            https://www.pyimagesearch.com/2018/02/19/real-time-object-detection-on-the-raspberry-pi-with-the-movidius-ncs/

      All tutorials work with 'imutils' package, which contains a benchmark tools.
      I achieved a 4.7 FPS with 'graph' converted from pretrained MobileNet model

## Building the flask web platform:
* The platform will be able to Stream camera video feed from the raspberry.
* The web platform will have the following interface:
  (no calculation performed on the web platform)
  * video feed window (optional: add gimbal position gui).
  * buttons for controlling the gimbal.
  * button for toggling smart_mode / simple_mode.
  * button for toggling auto_movement (enabled only in smart mode).
  * botton for pet tracking toggle: Tom / Mufasa (enabled if auto_movement is ON)


## Building the camera gimbal module.
* Allow movement in 3 axes.
* Send gimbal position as feedback


# ___TEMPORARY README, work in progress___
  # Flask video streamer
  Video streamer with image processing capabilities.


  Environment:
  * ubuntu 18.4
  * python 3.6.6

  Dependency libraries:
  * Imutils 0.5.1
  * OpenCV 3.4.1
  * Flask 1.0.2


  # Usage:

  1. make sure you have all dependency libraries.
    for great openCV installation tutorial refer to:
    https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/

  2. for Flask tutorials, and installation refer to:
    http://flask.pocoo.org/

  # Tanks to:
  * OpenCV - for the greatest computer Vision library in this world ( and others)

  * Adrian and his crew at - https://www.pyimagesearch.com/ for the best How-to's tutorials
    and email support.

  *Flask, jusk the best web development platform EVER!

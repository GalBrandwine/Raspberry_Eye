# Raspberry Eye is (smart ip camera) platform for deep learning models implementations 
By accomplishing an IoT smart ip camera, i can now design new Newral network models, train, and compile them to 'graph's for working with tha Movidius ncs

This project dedicated to Tom & Mufasa, my pets :)


## Intro:
After learning Deep Learning from scratch ([DL4CV amazing book](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)), and experiencing with Image Processing projects ([image_tracking](https://youtu.be/GF0xc0aUvpI), [helloPy](https://github.com/GalBrandwine/HalloPy)) , i’ve decided it is time to combine them all together.


## The goal:
To build Smart IP camera that will recognize my pets, that is gimbalised and controllable via a web interface (simple)


## Project components:
* A trained Neural Network model.
* A raspberry pi 3 B+.
* Movidius NCS.
* 3 servos.
* USB camera.


## Project Requirement specifications:
* OOP driven
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
      
  * Note:
    * Most of the work i did on my laptop, and then git-pulled to raspberry. in order to successfully ssh to the rasbperry,
    ive needed to prepare the [cd card for auto connecting to the wifi](https://www.raspberrypi.org/forums/viewtopic.php?t=191252).
    
    
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
* Allow 3 dof movement.
* Send gimbal position as feedback


# ___TEMPORARY README, work in progress:___
  
  # For building a client/server opencv streammer with tcp[Take a look at this tutorial](https://www.pyimagesearch.com/2019/04/15/live-video-streaming-over-network-with-opencv-and-imagezmq/?__s=hpwvpwmauc3ghqot4qpa)
   this project can get commands down the stream, im sure!
  
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

# from RPIO import PWM
import logging
import time
from flask import Flask
from flask import render_template
from flask import Response
from flask import g
from raspberry_eye.shared.flag_holder import FlagsHolder
from raspberry_eye.high_level_camera import CameraModule
from raspberry_eye.smart_camera import SmartCamera

# Create loggers.
camera_logger = logging.getLogger('camera_handler')
ch = logging.StreamHandler()
# create formatter and add it to the handlers.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to loggers.
camera_logger.addHandler(ch)

local_fh = FlagsHolder()
app = Flask(__name__)




# This function maps the angle we want to move the servo to, to the needed PWM value
def angleMap(angle):
    return int((round((1950.0 / 180.0), 0) * angle) + 550)


# Create a dictionary called pins to store the pin number, name, and angle
pins = {
    23: {'name': 'pan', 'angle': 90},
    22: {'name': 'tilt', 'angle': 90}
}


# Create two servo objects using the RPIO PWM library
# servoPan = PWM.Servo()
# servoTilt = PWM.Servo()

# Setup the two servos and turn both to 90 degrees
# servoPan.set_servo(23, angleMap(90))
# servoPan.set_servo(22, angleMap(90))

# Cleanup any open objects
# def cleanup():
#     servo.stop_servo(23)
#     servo.stop_servo(22)

# Load the main form template on web request for the root page
@app.route('/', methods=('GET', 'POST'))
def main():
    timeNow = time.asctime(time.localtime(time.time()))
    # Create a template data dictionary to send any data to the template
    template_data = {
        'time': timeNow,
        'title': 'PiCam_my'
    }
    # Pass the template data into the template picam.html and return it to the user
    return render_template('picam.html', **template_data)



def gen(camera):
    """Get frame from stream and preprocess before posting it on line. """

    #temp_smart_camera_toggle = local_fh.smart_toggle_global
    #temp_tom_toggle = local_fh.tom_toggle
    #temp_mufasa_toggle = local_fh.mufasa_toggle
    camera.capture("person")
    
    while True:

        # Update tom, mufasa, and smart toggles:
        # params: smart_on, smart_off (simple), cat, dog
        #if temp_smart_camera_toggle is not local_fh.smart_toggle_global:
            ## Smart toggle has changed.
            #temp_smart_camera_toggle = local_fh.smart_toggle_global
            #if temp_smart_camera_toggle is True:
                #camera.toggle_camera_modes('smart_on')
            #else:
                #camera.toggle_camera_modes('smart_off')

        ## For tom, mufasa and other future objects, toggle them on ONLY if 'smart' is ON.
        #if temp_tom_toggle is not local_fh.tom_toggle:
            #temp_tom_toggle = local_fh.tom_toggle
            #if temp_smart_camera_toggle is True and temp_tom_toggle is True:
                #camera.toggle_camera_modes('cat')
            #elif temp_smart_camera_toggle is True and temp_tom_toggle is False and temp_mufasa_toggle is False:
                #camera.toggle_camera_modes('smart_on')

        #if temp_mufasa_toggle is not local_fh.mufasa_toggle:
            #temp_mufasa_toggle = local_fh.mufasa_toggle
            #if temp_smart_camera_toggle is True and temp_mufasa_toggle is True:
                #camera.toggle_camera_modes('dog')
            #elif temp_smart_camera_toggle is True and temp_tom_toggle is False and temp_mufasa_toggle is False:
                #camera.toggle_camera_modes('smart_on')

        #if local_fh.release_camera is True:
            #local_fh.release_camera = False
            #camera.release()
            #break

        frame = camera.read()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    


@app.route('/video_feed')
def video_feed():
    """Entering this url, initiate VideoCamera object, that'll capture images from camera
     and stream them back to web-page.
     """

    # TODO: add global flag for determining which camera will be deployed: simple or smart.
    #return Response(gen(CameraModule(0, camera_logger)),
    #                mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(gen(SmartCamera("/home/pi/workspace/Raspberry_Eye/graphs/mobilenetgraph", 0)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# The function below is executed when someone requests a URL with a move direction
@app.route("/move/<direction>")
def move(direction):
    # Choose the direction of the request
    print("In direction: {}".format(direction))
    if direction == 'left':
        # Increment the angle by 10 degrees
        na = pins[23]['angle'] + 10
        # Verify that the new angle is not too great
        if int(na) <= 180:
            # Change the angle of the servo
            # servoPan.set_servo(23, angleMap(na))
            # Store the new angle in the pins dictionary
            pins[23]['angle'] = na
        return str(na) + ' ' + str(angleMap(na))
    elif direction == 'right':
        na = pins[23]['angle'] - 10
        if na >= 0:
            # servoPan.set_servo(23, angleMap(na))
            pins[23]['angle'] = na
        return str(na) + ' ' + str(angleMap(na))
    elif direction == 'up':
        print()
        na = pins[22]['angle'] + 10
        if na <= 180:
            # servoTilt.set_servo(22, angleMap(na))
            pins[22]['angle'] = na
        return str(na) + ' ' + str(angleMap(na))
    elif direction == 'down':
        na = pins[22]['angle'] - 10
        if na >= 0:
            # servoTilt.set_servo(22, angleMap(na))
            pins[22]['angle'] = na
        return str(na) + ' ' + str(angleMap(na))

    return "Pressed"


# The function below is executed when someone requests a URL with a smart_toggle button pressed
@app.route("/smart_toggle/<smart_toggle_input>")
def smart_toggle(smart_toggle_input):
    message = "In smart_toggle. {}".format(smart_toggle_input)
    # Toggle smart_toggle_global
    if smart_toggle_input == "smart_on":
        local_fh.smart_toggle_global = True
        message = message + " is now {}".format(local_fh.smart_toggle_global)

    if smart_toggle_input == "smart_off":
        local_fh.smart_toggle_global = False
        local_fh.tom_toggle = False
        local_fh.mufasa_toggle = False
        message = message + " is now {}".format(local_fh.smart_toggle_global)

    if smart_toggle_input == "tom_smart_on":
        local_fh.tom_toggle = True
        local_fh.mufasa_toggle = False
        message = message + " is now {}".format(local_fh.tom_toggle)
    if smart_toggle_input == "tom_smart_off":
        local_fh.tom_toggle = False
        message = message + " is now {}".format(local_fh.tom_toggle)

    if smart_toggle_input == "mufasa_smart_on":
        local_fh.mufasa_toggle = True
        local_fh.tom_toggle = False
        message = message + " is now {}".format(local_fh.mufasa_toggle)
    if smart_toggle_input == "mufasa_smart_off":
        local_fh.mufasa_toggle = False
        message = message + " is now {}".format(local_fh.mufasa_toggle)

    if smart_toggle_input == "release_camera":
        local_fh.release_camera = True
        message = message + " is now {}".format(local_fh.release_camera)
    print(message)
    return "Pressed"


# Function to manually set a motor to a specific pluse width
@app.route("/<motor>/<pulsewidth>")
def manual(motor, pulsewidth):
    if motor == "pan":
        print("pan")
        # servoPan.set_servo(23, int(pulsewidth))
    elif motor == "tilt":
        print("tilt")
        # servoTilt.set_servo(22, int(pulsewidth))
    return "Moved"



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)  # todo: remove debug when done

"""For running vie terminal:

cd ~/../../Raspberry_Eye/
type into terminal: FLASK_APP=rasbperry_eye/raspberry_eye_flask.py flask run.
"""

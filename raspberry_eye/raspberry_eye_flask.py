# from RPIO import PWM
from flask import Flask
from flask import render_template
from flask import Response
from utils.camera import VideoCamera

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
    # Create a template data dictionary to send any data to the template
    template_data = {
        'title': 'PiCam_my'
    }
    # Pass the template data into the template picam.html and return it to the user
    return render_template('picam.html', **template_data)


def gen(camera):
    """Get frame from stream and preprocess before posting it on line. """
    while True:
        frame = camera.get_frame()

        # TODO: add cat_detector

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    """Entering this url, initiate VideoCamera object, that'll capture images from camera
     and stream them back to web-page.
     """
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# The function below is executed when someone requests a URL with a move direction
@app.route("/<direction>")
def move(direction):
    # Choose the direction of the request
    print(direction)
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

    return "Prassed"


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


# Clean everything up when the app exits
# atexit.register(cleanup)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
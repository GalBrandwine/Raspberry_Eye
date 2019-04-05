from flask import Flask, render_template, Response
from raspberry_eye.simple_camera import SimpleCamera

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('picam.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    """Entering this url, initiate VideoCamera object, that'll capture images from camera
     and stream them back to web-page.
     """
    return Response(gen(SimpleCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
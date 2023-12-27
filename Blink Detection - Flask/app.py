from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import threading
from imutils import face_utils
import dlib

app = Flask(__name__)
socketio = SocketIO(app)

cap = cv2.VideoCapture(0)
lock = threading.Lock()

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def generate_frames():
    global cap, lock
    blink_counter = 0
    blink_in_progress = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        with lock:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for (i, rect) in enumerate(rects):
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[42:48]
                right_eye = shape[36:42]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)

                avg_ear = (left_ear + right_ear) / 2

                if avg_ear < 0.2:
                    if not blink_in_progress:
                        blink_in_progress = True
                        socketio.emit('blink_message', {'message': 'Blink detected! Total Blinks: {}'.format(blink_counter)})
                        print("Blink in progress!")
                else:
                    if blink_in_progress:
                        blink_in_progress = False
                        blink_counter += 1
                        socketio.emit('blink_message', {'message': 'Blink detected! Total Blinks: {}'.format(blink_counter)})
                        print("Blink detected! Total Blinks:", blink_counter)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
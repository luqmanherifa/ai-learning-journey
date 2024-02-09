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

face_cascPath = "src\\face.xml"
eye_cascPath = "src\\eye.xml"
nose_cascPath = "src\\nose.xml"
smile_cascPath = "src\\smile.xml"

faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)
noseCascade = cv2.CascadeClassifier(nose_cascPath)
smileCascade = cv2.CascadeClassifier(smile_cascPath)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def generate_frames():
    global cap, lock

    while True:
        success, frame = cap.read()
        if not success:
            break

        with lock:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for (i, rect) in enumerate(rects):
                cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0), 2)
                roi_gray = gray[rect.top():rect.bottom(), rect.left():rect.right()]
                roi_color = frame[rect.top():rect.bottom(), rect.left():rect.right()]

                eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))
                # Eye detection
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    eye_symmetry = np.abs(ex - (ex + ew)) if len(eyes) > 1 else -1
                    cv2.putText(frame, f'Eye Symmetry: {eye_symmetry:.2f}', (rect.left(), rect.top() - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                noses = noseCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))
                # Nose detection
                for (nx, ny, nw, nh) in noses:
                    cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (128, 0, 128), 2)
                    nose_symmetry = np.abs(nx - (nx + nw)) if len(noses) > 0 else -1
                    cv2.putText(frame, f'Nose Symmetry: {nose_symmetry:.2f}', (rect.left(), rect.top() - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
                # Smile detection
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
                    smile_symmetry = np.abs(sx - (sx + sw)) if len(smiles) > 0 else -1
                    cv2.putText(frame, f'Smile Symmetry: {smile_symmetry:.2f}', (rect.left(), rect.top()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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

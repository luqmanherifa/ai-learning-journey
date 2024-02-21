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

# Inisialisasi detektor wajah dan prediktor landmark dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inisialisasi detektor cascade untuk deteksi mata, hidung, dan senyum
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

def detect_blink(shape):
    left_eye = shape[42:48]
    right_eye = shape[36:42]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    avg_ear = (left_ear + right_ear) / 2

    return avg_ear

def calculate_total_average_symmetry(eye_symmetry, nose_symmetry, smile_symmetry):
    total_symmetry = (eye_symmetry + nose_symmetry + smile_symmetry) / 3
    return total_symmetry

def generate_frames():
    global cap, lock, blink_counter, blink_in_progress

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

                # Deteksi landmark wajah menggunakan dlib
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Tambahkan tindakan apa pun yang Anda inginkan dengan landmark di sini
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Deteksi mata, hidung, dan senyum menggunakan cascade classifier
                eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))
                total_eye_symmetry = sum([np.abs(ew - eh) for (_, _, ew, eh) in eyes]) / len(eyes) if len(eyes) > 0 else -1

                noses = noseCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))
                total_nose_symmetry = sum([np.abs(nw - nh) for (_, _, nw, nh) in noses]) / len(noses) if len(noses) > 0 else -1

                smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
                total_smile_symmetry = sum([np.abs(sw - sh) for (_, _, sw, sh) in smiles]) / len(smiles) if len(smiles) > 0 else -1

                total_avg_symmetry = calculate_total_average_symmetry(total_eye_symmetry, total_nose_symmetry, total_smile_symmetry)
                cv2.putText(frame, f'Total Average Symmetry: {total_avg_symmetry:.2f}', (rect.left(), rect.top() - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    eye_symmetry = np.abs(ew - eh) if len(eyes) > 1 else -1
                    cv2.putText(frame, f'Eye Symmetry: {eye_symmetry:.2f}', (rect.left(), rect.top() - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Deteksi kedipan
                ear = detect_blink(shape)
                if ear < 0.2:
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

                for (nx, ny, nw, nh) in noses:
                    cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (128, 0, 128), 2)
                    nose_symmetry = np.abs(nw - nh) if len(noses) > 0 else -1
                    cv2.putText(frame, f'Nose Symmetry: {nose_symmetry:.2f}', (rect.left(), rect.top() - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
                    smile_symmetry = np.abs(sw - sh) if len(smiles) > 0 else -1
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
    blink_counter = 0
    blink_in_progress = False
    socketio.run(app, debug=True)

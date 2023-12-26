# Import library yang dibutuhkan
from flask import Flask, render_template, Response
import cv2

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Inisialisasi objek deteksi mata, hidung, dan senyum OpenCV
eye_cascade = cv2.CascadeClassifier('src/eye.xml')  # Sesuaikan dengan lokasi file eye.xml
nose_cascade = cv2.CascadeClassifier('src/nose.xml')  # Sesuaikan dengan lokasi file nose.xml
smile_cascade = cv2.CascadeClassifier('src/smile.xml')  # Sesuaikan dengan lokasi file smile.xml

# Fungsi untuk melakukan deteksi mata, hidung, dan senyum pada frame
def detect_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Detect nose in the frame
    noses = nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))
    for (nx, ny, nw, nh) in noses:
        cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (128, 0, 128), 2)

    # Detect smile in the frame
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)

    return frame

# Fungsi untuk mengambil frame dari kamera
def generate_frames():
    cap = cv2.VideoCapture(0)  # Ganti angka 0 dengan alamat IP kamera jika menggunakan kamera eksternal

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_features(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk streaming video dengan deteksi mata, hidung, dan senyum
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)

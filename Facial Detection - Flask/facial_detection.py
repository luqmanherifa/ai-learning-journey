# Import library yang dibutuhkan
from flask import Flask, render_template, Response
import cv2  # Mengimport library OpenCV untuk pemrosesan gambar

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Inisialisasi objek deteksi mata, hidung, dan senyum OpenCV
eye_cascade = cv2.CascadeClassifier('src/eye.xml')  # Menggunakan model deteksi mata
nose_cascade = cv2.CascadeClassifier('src/nose.xml')  # Menggunakan model deteksi hidung
smile_cascade = cv2.CascadeClassifier('src/smile.xml')  # Menggunakan model deteksi senyum

# Fungsi untuk melakukan deteksi mata, hidung, dan senyum pada frame
def detect_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengubah gambar menjadi skala abu-abu untuk meningkatkan kinerja deteksi

    # Detect mata dalam frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)  # Menggambar kotak di sekitar mata yang terdeteksi

    # Detect hidung dalam frame
    noses = nose_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))
    for (nx, ny, nw, nh) in noses:
        cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (128, 0, 128), 2)  # Menggambar kotak di sekitar hidung yang terdeteksi

    # Detect senyum dalam frame
    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)  # Menggambar kotak di sekitar senyum yang terdeteksi

    return frame

# Fungsi untuk mengambil frame dari kamera
def generate_frames():
    cap = cv2.VideoCapture(0)  # Ganti angka 0 dengan alamat IP kamera jika menggunakan kamera eksternal

    while True:
        success, frame = cap.read()  # Membaca frame dari kamera
        if not success:
            break
        else:
            frame = detect_features(frame)  # Memanggil fungsi deteksi mata, hidung, dan senyum
            ret, buffer = cv2.imencode('.jpg', frame)  # Mengonversi frame menjadi format gambar JPEG
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Menghasilkan frame untuk streaming

    cap.release()

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')  # Menampilkan halaman utama

# Route untuk streaming video dengan deteksi mata, hidung, dan senyum
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # Menggunakan Response untuk streaming video

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)  # Menjalankan aplikasi Flask dalam mode debug

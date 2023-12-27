# Mengimpor modul dan pustaka yang diperlukan
from imutils import face_utils  # Menggunakan fungsi bantuan untuk bekerja dengan struktur wajah
import dlib  # Digunakan untuk deteksi wajah dan penempatan landmark
import cv2  # Menggunakan OpenCV untuk pemrosesan gambar
import numpy as np  # Menggunakan NumPy untuk operasi numerik

# Mendefinisikan path untuk model prediksi landmark wajah
p = "shape_predictor_68_face_landmarks.dat"

# Menginisialisasi detektor wajah dan predictor landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Membuka koneksi dengan kamera (0 untuk kamera default)
cap = cv2.VideoCapture(0)

# Mendefinisikan fungsi perbandingan aspek rasio mata
def eye_aspect_ratio(eye):
    # Menghitung panjang sisi mata
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Menghitung panjang garis tengah mata
    C = np.linalg.norm(eye[0] - eye[3])

    # Menghitung dan mengembalikan aspek rasio mata (eye aspect ratio)
    ear = (A + B) / (2.0 * C)
    return ear

# Inisialisasi variabel untuk menghitung kedipan mata
blink_counter = 0
blink_in_progress = False

# Loop utama untuk membaca dan memproses setiap frame dari kamera
while True:
    # Membaca frame dari kamera
    _, image = cap.read()
    # Mengubah gambar ke skala abu-abu untuk deteksi wajah lebih baik
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah dalam frame
    rects = detector(gray, 0)

    # Loop untuk setiap wajah yang terdeteksi
    for (i, rect) in enumerate(rects):
        # Menghitung landmark wajah
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Mengambil koordinat mata kiri dan kanan
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        # Menghitung aspek rasio mata kiri dan kanan
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Menghitung rata-rata aspek rasio mata
        avg_ear = (left_ear + right_ear) / 2

        # Mengecek jika mata tertutup (kedipan)
        if avg_ear < 0.2:
            if not blink_in_progress:
                blink_in_progress = True
                print("Blink in progress!")
        else:
            if blink_in_progress:
                blink_in_progress = False
                blink_counter += 1
                print("Blink detected! Total Blinks:", blink_counter)

        # Menampilkan landmark wajah pada gambar
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # Menampilkan gambar hasil dengan landmark dan deteksi kedipan mata
    cv2.imshow("Output", image)

    # Menunggu tombol ESC untuk keluar dari loop
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Menutup jendela tampilan dan melepaskan kamera
cv2.destroyAllWindows()
cap.release()

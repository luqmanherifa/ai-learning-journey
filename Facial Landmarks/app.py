# Mengimpor modul yang diperlukan
from imutils import face_utils  # imutils membantu dalam pengolahan gambar
import dlib  # dlib menyediakan algoritma pembaca wajah
import cv2  # cv2 adalah OpenCV, digunakan untuk pemrosesan gambar

# Path untuk model prediksi landmark wajah
p = "shape_predictor_68_face_landmarks.dat"

# Inisialisasi pembaca wajah dan prediktor landmark wajah
detector = dlib.get_frontal_face_detector()  # detektor wajah frontal
predictor = dlib.shape_predictor(p)  # prediktor landmark wajah

# Inisialisasi kamera (0 untuk kamera default)
cap = cv2.VideoCapture(0)

# Loop utama untuk membaca dan memproses setiap frame
while True:
    _, image = cap.read()  # Membaca frame dari kamera
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Mengubah ke skala abu-abu

    rects = detector(gray, 0)  # Mendeteksi wajah pada frame

    # Loop melalui wajah yang terdeteksi
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)  # Mendapatkan landmark wajah
        shape = face_utils.shape_to_np(shape)  # Mengubah landmark menjadi array NumPy

        # Loop melalui setiap landmark dan menambahkan lingkaran
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Menambahkan lingkaran ke setiap landmark

    cv2.imshow("Output", image)  # Menampilkan hasil
    k = cv2.waitKey(5) & 0xFF  # Menunggu tombol ESC untuk keluar
    if k == 27:
        break

cv2.destroyAllWindows()  # Menutup jendela OpenCV
cap.release()  # Melepas kamera

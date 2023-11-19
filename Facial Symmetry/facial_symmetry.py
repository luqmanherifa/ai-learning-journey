import cv2
import numpy as np

# Path to the cascade XML files
face_cascPath = "src\\face.xml"
eye_cascPath = "src\\eye.xml"
nose_cascPath = "src\\nose.xml"
smile_cascPath = "src\\smile.xml"

# Load cascades
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)
noseCascade = cv2.CascadeClassifier(nose_cascPath)
smileCascade = cv2.CascadeClassifier(smile_cascPath)

# Open the camera
video_capture = cv2.VideoCapture(0)

# Inisialisasi array untuk menyimpan skor simetri
eye_symmetry_scores = []
nose_symmetry_scores = []
smile_symmetry_scores = []

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Region of Interest (ROI) for eyes, nose, and smile within the face rectangle
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face
        eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))

        cv2.putText(frame, 'Number of Eyes Detected: {}'.format(len(eyes)), (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if len(eyes) > 1:
            eyes_x_positions = [ex for (ex, _, _, _) in eyes]
            eye_symmetry = np.abs(eyes_x_positions[0] - eyes_x_positions[1])
            cv2.putText(frame, f'Eye Symmetry: {eye_symmetry:.2f}', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            eye_symmetry_scores.append(eye_symmetry)

            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes[:2]
            if eye_symmetry > 100:
                cv2.rectangle(roi_color, (ex1, ey1), (ex1+ew1, ey1+eh1), (0, 0, 255), 2)
                cv2.rectangle(roi_color, (ex2, ey2), (ex2+ew2, ey2+eh2), (0, 0, 255), 2)
            else:
                cv2.rectangle(roi_color, (ex1, ey1), (ex1+ew1, ey1+eh1), (0, 255, 0), 2)
                cv2.rectangle(roi_color, (ex2, ey2), (ex2+ew2, ey2+eh2), (0, 255, 0), 2)
            print(f'Eye Symmetry - Left: {eye_symmetry:.2f}, Right: {eye_symmetry:.2f}')
        elif len(eyes) == 1:
            cv2.putText(frame, 'Need more than 1 eye for symmetry calculation', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            (ex, ey, ew, eh) = eyes[0]
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            print('Only one eye detected, symmetry calculation not possible')
        else:
            cv2.putText(frame, 'Eyes Not Detected', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Detect nose in the face
        noses = noseCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))
        if len(noses) > 0:
            cv2.rectangle(roi_color, (noses[0][0], noses[0][1]), (noses[0][0]+noses[0][2], noses[0][1]+noses[0][3]), (128, 0, 128), 2)
            nose_symmetry = np.abs(noses[0][0] - (noses[0][0] + noses[0][2]))
            cv2.putText(frame, f'Nose Symmetry: {nose_symmetry:.2f}', (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            nose_symmetry_scores.append(nose_symmetry)
            if nose_symmetry > 70:
                cv2.rectangle(roi_color, (noses[0][0], noses[0][1]), (noses[0][0]+noses[0][2], noses[0][1]+noses[0][3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(roi_color, (noses[0][0], noses[0][1]), (noses[0][0]+noses[0][2], noses[0][1]+noses[0][3]), (255, 0, 255), 2)
            print(f'Nose Symmetry: {nose_symmetry:.2f}')
        else:
            cv2.putText(frame, 'Nose Not Detected', (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Detect smile in the face
        smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        if len(smiles) > 0:
            cv2.rectangle(roi_color, (smiles[0][0], smiles[0][1]), (smiles[0][0]+smiles[0][2], smiles[0][1]+smiles[0][3]), (0, 255, 255), 2)
            smile_symmetry = np.abs(smiles[0][0] - (smiles[0][0] + smiles[0][2]))
            cv2.putText(frame, f'Smile Symmetry: {smile_symmetry:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            smile_symmetry_scores.append(smile_symmetry)
            if smile_symmetry > 120:
                cv2.rectangle(roi_color, (smiles[0][0], smiles[0][1]), (smiles[0][0]+smiles[0][2], smiles[0][1]+smiles[0][3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(roi_color, (smiles[0][0], smiles[0][1]), (smiles[0][0]+smiles[0][2], smiles[0][1]+smiles[0][3]), (0, 255, 255), 2)
            print(f'Smile Symmetry: {smile_symmetry:.2f}')
        else:
            cv2.putText(frame, 'Smile Not Detected', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'esc' is pressed
    if cv2.waitKey(1) == 27:  # 27 is the ASCII code for the 'esc' key
        break

# Menghitung rata-rata dari skor simetri
average_eye_symmetry = np.mean(eye_symmetry_scores)
average_nose_symmetry = np.mean(nose_symmetry_scores)
average_smile_symmetry = np.mean(smile_symmetry_scores)

# Menampilkan rata-rata skor simetri pada konsol
print(f'Average Eye Symmetry: {average_eye_symmetry:.2f}')
print(f'Average Nose Symmetry: {average_nose_symmetry:.2f}')
print(f'Average Smile Symmetry: {average_smile_symmetry:.2f}')

# Release the camera
video_capture.release()
cv2.destroyAllWindows()

import cv2

# Path to the cascade XML file
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

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Region of Interest (ROI) for eyes and smile within the face rectangle
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face
        eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Detect nose in the face
        noses = noseCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), maxSize=(100, 100))
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (128, 0, 128), 2)

        # Detect smile in the face
        smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'esc' is pressed
    if cv2.waitKey(1) == 27:  # 27 is the ASCII code for the 'esc' key
        break

# Release the camera
video_capture.release()
cv2.destroyAllWindows()
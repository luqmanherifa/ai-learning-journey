import cv2
import numpy as np

# Path to the cascade XML file
face_cascPath = "src\\face.xml"

# Load cascade
faceCascade = cv2.CascadeClassifier(face_cascPath)

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

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Calculate the symmetry score
        symmetry_score = np.abs(x - (x+w))

        # Change the color of the face rectangle to red if symmetry is large
        face_color = (0, 0, 255) if symmetry_score > 250 else (255, 0, 0)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 2)

        # Calculate the center of the face
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw a crosshair at the center of the face
        cv2.line(frame, (center_x, y), (center_x, y + h), (0, 255, 0), 2)
        cv2.line(frame, (x, center_y), (x + w, center_y), (0, 255, 0), 2)

        # Display the symmetry score
        cv2.putText(frame, f'Symmetry: {symmetry_score:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'esc' is pressed
    if cv2.waitKey(1) == 27:  # 27 is the ASCII code for the 'esc' key
        break

# Release the camera
video_capture.release()
cv2.destroyAllWindows()

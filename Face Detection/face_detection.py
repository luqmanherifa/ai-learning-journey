import cv2

# Path to the cascade XML file
face_cascPath = "src\\face.xml"

# Load the face cascade
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

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'esc' is pressed
    if cv2.waitKey(1) == 27:  # 27 is the ASCII code for the 'esc' key
        break

# Release the camera
video_capture.release()
cv2.destroyAllWindows()

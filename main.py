import cv2
import dlib


cap = cv2.VideoCapture(0)  
face_detector = dlib.get_frontal_face_detector()

tracker = cv2.TrackerCSRT_create()

# Flag to indicate if the tracker is initialized
tracker_initialized = False

while True:
    ret, frame = cap.read()

    # Check if the frame is properly captured
    if not ret:
        print("Failed to capture frame from the camera.")
        break
    # Convert the frame to RGB (required by dlib)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not tracker_initialized:
        # Perform face detection using dlib
        faces = face_detector(rgb, 1)

        # If a face is detected, initialize the tracker
        if len(faces) > 0:
            # Select the first detected face
            face = faces[0]
            
            # Convert the face coordinates to OpenCV format
            (x, y, w, h) = (face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top())
            
            # Initialize the tracker with the selected face region
            tracker.init(frame, (x, y, w, h))
            tracker_initialized = True
    else:
        # Update the tracker
        success, bbox = tracker.update(frame)

        # If tracking is successful, draw the bounding box
        if success:
            (x, y, w, h) = [int(coord) for coord in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # If tracking fails, reset the tracker
            tracker_initialized = False

    cv2.imshow('Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


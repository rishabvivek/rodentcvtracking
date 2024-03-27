import cv2
import dlib

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use the appropriate camera index

# Load the dlib face detector
face_detector = dlib.get_frontal_face_detector()

# Initialize the CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Flag to indicate if the tracker is initialized
tracker_initialized = False

while True:
    # Capture frame from the camera
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

    # Display the output frame
    cv2.imshow('Face Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera resources
cap.release()
cv2.destroyAllWindows()


# import cv2

# # Initialize the camera
# cap = cv2.VideoCapture(0)  # Use the appropriate camera index

# # Load the Haar cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize the tracker (e.g., KCF tracker)
# tracker = cv2.TrackerKCF_create()

# # Flag to indicate if the tracker is initialized
# tracker_initialized = False

# while True:
#     # Capture frame from the camera
#     ret, frame = cap.read()

#     # Check if the frame is properly captured
#     if not ret:
#         print("Failed to capture frame from the camera.")
#         break

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     if not tracker_initialized:
#         # Perform face detection
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#         # If a face is detected, initialize the tracker
#         if len(faces) > 0:
#             # Select the first detected face
#             (x, y, w, h) = faces[0]
            
#             # Initialize the tracker with the selected face region
#             tracker.init(frame, (x, y, w, h))
#             tracker_initialized = True
#     else:
#         # Update the tracker
#         success, bbox = tracker.update(frame)

#         # If tracking is successful, draw the bounding box
#         if success:
#             (x, y, w, h) = [int(coord) for coord in bbox]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         else:
#             # If tracking fails, reset the tracker
#             tracker_initialized = False

#     # Display the output frame
#     cv2.imshow('Face Tracking', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera resources
# cap.release()
# cv2.destroyAllWindows()
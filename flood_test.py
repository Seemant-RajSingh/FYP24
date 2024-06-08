import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from ultralytics import YOLO

model1 = YOLO('yolov8n.pt')

# Load the saved flooding detection model
model = load_model("fine_tuned_flood_detection_model_1")

def preprocess_image(img):
    img_array = image.img_to_array(cv2.resize(img, (224, 224)))
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
#     #
video_path = 'flood_stuck_1.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(0)
person_class_id = 0
screen_width = 720  # Adjust these dimensions according to your screen resolution
screen_height = 500
classes_to_detect = [0, 2]
person_count=0

while True:
    ret, frame = cap.read()  # Read a frame from the video stream

    if not ret:
        break  # Break the loop if there are no more frames

    results = model1.track(frame, conf=0.11, show=True,classes=classes_to_detect)

    frame_ = results[0].plot()
    person_count=0
    for r in results:
        print(len(r))
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            # print(cls)
            if (cls==0):
                person_count=person_count+1

    frame_ = results[0].plot()

    frame_ = cv2.resize(frame_, (screen_width, screen_height))

    preprocessed_image = preprocess_image(frame)
    predictions = model.predict(preprocessed_image)
    result = np.argmax(predictions)
    labels = ['Flooding', 'No Flooding']
    predicted_label = labels[result]
    print(predicted_label)

    if predicted_label == 'Flooding':
        cv2.putText(frame_, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame_, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame_, f'Person: {person_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
    # Display the frame with detections
    cv2.imshow('Detection Output', frame_)

    if 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()


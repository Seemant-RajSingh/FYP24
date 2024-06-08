# Prepare image for mobilenet prediction
# from IPython.display import Image
# from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Activation
# mobile = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False)
# # mobile.summary()
# # Store all layers of the original mobilenet except the last 5 layers in variable x
# # There is no predefined logic behind this, it just gives the optimal results for this task
# # Also, we will be only training the last 12 layers of the mobilenet during finetuning as we want
# # it to keep all of the previously learned weightsda
# x = mobile.layers[-12].output
#
# # Create global pooling, dropout and a binary output layer, as we want our model to be a binary classifier,
# # i.e. to classify flooding and no flooding
# x = keras.layers.GlobalAveragePooling2D()(x)
# x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
# output = Dense(units=2, activation='sigmoid')(x)
# model = Model(inputs=mobile.input, outputs=output)
#
# def preprocess_image(file):
#     img_path = 'D:/project/collage/flood/evaluate/'
#     img = image.load_img(img_path + file, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array_expanded_dims = np.expand_dims(img_array, axis=0)
#     return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
# # Display image which we want to predict
#
# Image(filename='D:/project/collage/flood/evaluate/8.jpg', width=300,height=200)
# # Preprocess image and make prediction
#
# preprocessed_image = preprocess_image('8.jpg')
# predictions = model.predict(preprocessed_image)
# # Print predicted accuracy scores for both classes, i.e. (1) Flooding, (2) No Flooding
# print(predictions)
# # Get the maximum probability score for predicted class from predictions array
# result = np.argmax(predictions)
# print(result)
# # Print the predicted class label
# labels = ['No Flooding', 'Flooding']
# print(labels[result])


# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import cv2
# from ultralytics import YOLO
# model1 = YOLO('yolov8n.pt')
# # Load the saved model
# model = load_model("fine_tuned_flood_detection_model")  # Replace with the path to your saved model
#
# # Define a function to preprocess the image
# def preprocess_image(file):
#     img_path = 'D:/project/collage/flood/evaluate/'
#     img = image.load_img(img_path + file, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array_expanded_dims = np.expand_dims(img_array, axis=0)
#     return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
#
# # Preprocess the image and make predictions
# img='5.jpg'
# preprocessed_image = preprocess_image(img)  # Replace '8.jpg' with your image filename
# predictions = model.predict(preprocessed_image)
#
# # Print predicted probabilities for each class
# print(predictions)
# results = model1.track(img.read(), conf=0.3, iou=0.5, show=True)  # model.track(frame, persist=True, classes=0)
# frame_ = results[0].plot()
#
# # out.write(frame_)
#
# # Get the predicted class label
# result = np.argmax(predictions)
# labels = ['Flooding', 'No Flooding']
# predicted_label = labels[result]
# print(predicted_label)
# while True:
#     cv2.imshow('Detection Output', frame_)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

import math
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from ultralytics import YOLO
import cvlib as cv
from cvlib.object_detection import draw_bbox
# Load the YOLOv5 model
model1 = YOLO('yolov8n.pt')

# Load the saved flooding detection model
model = load_model("fine_tuned_flood_detection_model")

# Read the image using OpenCV
# img_path = 'D:/project/collage/flood/evaluate/5.jpg'
# frame = cv2.imread(img_path)

# Define a function to preprocess the image
def preprocess_image(img):
    img_array = image.img_to_array(cv2.resize(img, (224, 224)))
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
#     #
video_path = 'flood_stuck_1.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)
person_class_id = 0
screen_width = 720  # Adjust these dimensions according to your screen resolution
screen_height = 500
classes_to_detect = [0, 2]
person_count=0

while True:
    ret, frame = cap.read()  # Read a frame from the video stream

    if not ret:
        break  # Break the loop if there are no more frames
    # bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov8n.pt')
    # im = draw_bbox(frame, bbox, label, conf)
    # Perform object detection using YOLOv5
    results = model1.track(frame, conf=0.11, show=True,classes=classes_to_detect)
    # result_car = model1.track(frame, conf=0.3, show=True, classes=2)


    # frame_ = result_car[0].plot()

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

            # if (currentClass == "car" and conf > 0.3):
            #     currentClass = classNames[cls]

            # if (currentClass == "person" and conf > 0.01) or (currentClass == "car" and conf > 0.3):
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
    frame_ = results[0].plot()
    # print(frame_)
    # person_count= len(results[0])
    frame_ = cv2.resize(frame_, (screen_width, screen_height))

    # Preprocess the frame and make predictions using the flooding detection model
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
# Preprocess the image and make predictions using the flooding detection model
# preprocessed_image = preprocess_image(frame)
# predictions = model.predict(preprocessed_image)
# result = np.argmax(predictions)
# labels = ['Flooding', 'No Flooding']
# predicted_label = labels[result]
# print(predicted_label)
#
# # Use YOLO for object detection
# results = model1.track(frame, conf=0.3, iou=0.5, show=True)
# frame_ = results[0].plot()
#
# while True:
#     cv2.imshow('Detection Output', frame_)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

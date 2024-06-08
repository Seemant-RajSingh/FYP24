import os
import shutil
import random
import itertools
# %matplotlib inline
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from keras import backend
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Activation
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input

labels = ['Flooding', 'No Flooding']
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)
#Loading pre-trained lightweight mobilenet image classifier
mobile = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False)
# mobile.summary()
# Store all layers of the original mobilenet except the last 5 layers in variable x
# There is no predefined logic behind this, it just gives the optimal results for this task
# Also, we will be only training the last 12 layers of the mobilenet during finetuning as we want
# it to keep all of the previously learned weightsda
x = mobile.layers[-12].output

# Create global pooling, dropout and a binary output layer, as we want our model to be a binary classifier,
# i.e. to classify flooding and no flooding
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
output = Dense(units=2, activation='sigmoid')(x)
model = Model(inputs=mobile.input, outputs=output)
for layer in model.layers[:-23]:
    layer.trainable = False
model.summary()

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          epochs=10,
          verbose=2
)

# Saving and loading our trained for future use

model.save("fine_tuned_flood_detection_model")
# model.load_weights('fine_tuned_flood_detection_model')

# Make predictions and plot confusion matrix to look how well our model performed in classifying
# flooding and no flooding images

test_labels = test_batches.classes
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
precision = precision_score(y_true=test_labels, y_pred=predictions.argmax(axis=1))
f1_score = f1_score(y_true=test_labels, y_pred=predictions.argmax(axis=1))
accuracy = accuracy_score(y_true=test_labels, y_pred=predictions.argmax(axis=1))
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
# Pring precision, F1 score and accuracy of our model
print('Precision: ', precision)
print('F1 Score: ', f1_score)
print('Accuracy: ', accuracy)
# Confusion Matrix
test_batches.class_indices
cm_plot_labels = ['Flooding','No Flooding']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')





#
#
# import cv2
# import cv2 as cv
# import numpy as np
# import math
# import argparse
# import imutils
# from collections import deque
#
# cap = cv2.VideoCapture(0)
#
# while (cap.isOpened()):
#     ret, img = cap.read()
#     rest, img1 = cap.read()
#     img = img[5:600, 0:43]
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     # cv2.imshow('input', blur)
#     contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     drawing = np.zeros(img.shape, np.uint8)
#     drawing = drawing[20:500, 22:24]
#     max_area = 0
#
#     for i in range(len(contours)):
#         cnt = contours[i]
#         area = cv2.contourArea(cnt)
#         if (area > max_area):
#             max_area = area
#             ci = i
#     cnt = contours[ci]
#     hull = cv2.convexHull(cnt)
#     moments = cv2.moments(cnt)
#     if moments['m00'] != 0:
#         cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
#         cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
#
#     centr = (cx, cy)
#     cv2.circle(drawing, centr, 1, [0, 0, 255], 2)
#     # cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
#     cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)
#
#     # construct the argument parse and parse the arguments
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-v", "--video",
#                     help="path to the (optional) video file")
#     ap.add_argument("-b", "--buffer", type=int, default=64,
#                     help="max buffer size")
#     args = vars(ap.parse_args())
#
#     greenLower = (0, 120, 70)
#     greenUpper = (180, 255, 255)
#     pts = deque(maxlen=args["buffer"])
#     # if a video path was not supplied, grab the reference
#     # to the webcam
#     if not args.get("video", False):
#         vs = drawing
#     # otherwise, grab a reference to the video file
#     else:
#         vs = cv2.VideoCapture(args["video"])
#
#     frame = drawing
#     # handle the frame from VideoCapture or VideoStream
#     frame = frame[1] if args.get("video", False) else frame
#     # if we are viewing a video and we did not grab a frame,
#     # then we have reached the end of the video
#     if frame is None:
#         break
#     # resize the frame, blur it, and convert it to the HSV
#     # color space
#     # frame = imutils.resize(frame, width=600)
#     blurred = cv2.GaussianBlur(frame, (11, 11), 0)
#     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#     cv2.imshow('input', hsv)
#     # construct a mask for the color "green", then perform
#     # a series of dilations and erosions to remove any small
#     # blobs left in the mask
#     mask = cv2.inRange(hsv, greenLower, greenUpper)
#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)
#
#     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
#                             cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     center = None
#     # only proceed if at least one contour was found
#     if len(cnts) > 0:
#         # find the largest contour in the mask, then use
#         # it to compute the minimum enclosing circle and
#         # centroid
#         c = max(cnts, key=cv2.contourArea)
#         ((x, y), radius) = cv2.minEnclosingCircle(c)
#         M = cv2.moments(c)
#         center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
#         centerx = int(M["m10"] / M["m00"])
#         centery = int(M["m01"] / M["m00"]) - int(radius)
#         if radius > 10:
#             # draw the circle and centroid on the frame,
#             # then update the list of tracked points
#             # cv2.circle(img1, (int(x), int(y)), int(radius),(0, 255, 255), 2)
#             cv2.circle(img1, (40, centery), 5, (255, 0, 0), -1)
#             cv2.putText(img1, "Water Elevation: " + str(centery), (40, centery - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (172, 90, 0), 2, cv2.LINE_AA, False)
#
#         # update the points queue
#     pts.appendleft(center)
#
#     for i in range(1, len(pts)):
#         # if either of the tracked points are None, ignore
#         # them
#         if pts[i - 1] is None or pts[i] is None:
#             continue
#         # otherwise, compute the thickness of the line and
#         # draw the connecting lines
#         thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
#         cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
#     # show the frame to our screen
#     # cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#     # if the 'q' key is pressed, stop the loop
#     if key == ord("q"):
#         break
#
#     cv2.imshow('output', img1)
#     # cv2.imshow('input', img)
#
#     k = cv2.waitKey(10)
#     if k == 27:
#         break

# import cv2
# from matplotlib import pyplot as plt
# import imutils
# import cv2
# import numpy as np
#
# # Read the image of the glass
# image = cv2.imread('glass2.jpg')  # Replace 'glass_image.jpg' with your image file
#
# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Apply Gaussian blur to reduce noise
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
# # Apply Canny edge detection
# edges = cv2.Canny(blurred, 50, 150)  # Adjust the thresholds as needed
# # cv2.imshow("edegs",edges)
#
#
# (T, bottle_threshold) = cv2.threshold(edges, 27.5, 255, cv2.THRESH_BINARY_INV)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# bottle_open = cv2.morphologyEx(bottle_threshold, cv2.MORPH_OPEN, kernel)
# cv2.imshow("Bottle Open 5 x 5", bottle_open)
#
# cv2.waitKey(0)
# # Find contours in the edgesed
# # find all contours
# contours = cv2.findContours(bottle_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours)
# bottle_clone = edges.copy()
# cv2.drawContours(bottle_clone, contours, -1, (255, 0, 0), 2)
# cv2.imshow("All Contours", bottle_clone)
# cv2.waitKey(0)
# areas = [cv2.contourArea(contour) for contour in contours]
# (contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a:a[1]))
# # print contour with largest area
# bottle_clone = edges.copy()
# cv2.drawContours(bottle_clone, [contours[-1]], -1, (255, 0, 0), 2)
# cv2.imshow("Largest contour", bottle_clone)
# cv2.waitKey(0)
#
#
#
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Draw the contours on the original image
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#
# # Display the results
# cv2.imshow('Canny Edge Detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

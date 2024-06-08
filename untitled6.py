# # from ultralytics import YOLO
# # import cv2
# #
# #
# # model = YOLO('yolov8n.pt')
# #
# # #video
# # # video_path = './test.mp4'
# #
# # # cap = cv2.VideoCapture(video_path)
# #
# # #real-time
# # cap = cv2.VideoCapture(0)
# #
# #
# # frame_width = int(cap.get(3))
# # frame_height = int(cap.get(4))
# # # out = cv2.VideoWriter("test_output.mp4", cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))
# #
# # ret = True
# # while ret:
# #     ret, frame = cap.read()
# #     if ret:
# #         results =model.track(frame, conf=0.3, iou=0.5, show=True)  #model.track(frame, persist=True, classes=0)
# #         frame_ = results[0].plot()
# #         cv2.imshow('Detection Output', frame_)
# #         # out.write(frame_)
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
#
#
# import cv2
# import imutils
#
# # Initializing the HOG person
# # detector
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#
# cap = cv2.VideoCapture('flood_stuck_2.mp4')
#
# while cap.isOpened():
#     # Reading the video stream
#     ret, image = cap.read()
#     if ret:
#         image = imutils.resize(image,
#                                width=min(400, image.shape[1]))
#
#         # Detecting all the regions
#         # in the Image that has a
#         # pedestrians inside it
#         (regions, _) = hog.detectMultiScale(image,
#                                             winStride=(4, 4),
#                                             padding=(4, 4),
#                                             scale=1.05)
#
#         # Drawing the regions in the
#         # Image
#         for (x, y, w, h) in regions:
#             cv2.rectangle(image, (x, y),
#                           (x + w, y + h),
#                           (0, 0, 255), 2)
#
#         # Showing the output Image
#         cv2.imshow("Image", image)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     else:
#         break
#
# cap.release()
# cv2.destroyAllWindows()


import numpy as np
from ultralytics import YOLO
import cv2
import math
# from sort import *
# from helper import create_video_writer

cap = cv2.VideoCapture("sample.mp4")  # For Video
# writer = create_video_writer(cap, "Output.mp4")
model = YOLO("yolov8n.pt")
# mask = cv2.imread("mask.png")
# Tracking
# tracker = Sort(max_age=20)

limitsDown = [150, 220, 250, 220]
limitsUp= [70, 170, 160, 170]
totalCountUp = []
totalCountDown = []
while True:
    success, img = cap.read()
    imgRegion = img
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            print(box)
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
 # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 6)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 , id= int(x1), int(y1), int(x2), int(y2) ,int(id)
        print(result)
        w, h = x2 - x1, y2 - y1

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Using cv2.putText() method
        image = cv2.putText(img, currentClass, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 0), 1, cv2.LINE_AA)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
                cv2.putText(img, f'Up: {len(totalCountUp)}', (480, 40), cv2.FONT_HERSHEY_PLAIN, 2, (139, 195, 75), 3)
                cv2.putText(img, f'Down: {len(totalCountDown)}', (480, 70), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 230), 3)
                cv2.imshow("Video", img)
                # writer.write(img)
                if cv2.waitKey(1) == ord("q"):
                    break
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
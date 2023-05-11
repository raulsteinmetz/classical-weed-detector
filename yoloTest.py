from roboflow import Roboflow
import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO('./dataset_yolov8/runs/detect/train8/weights/best.pt')

cap = cv2.VideoCapture("./videos/sample_video2.MP4")

i = 0
frame_counter = 0
# Loop through the frames of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    print(i)
    i += 1


    frame = cv2.resize(frame, (640, 640))

    results = model.predict(source=frame, save=False, save_txt=False)  # save predictions as labels

    for result in results:
        # Detection
        boxes = result.boxes.xyxy  # box with xywh format, (N, 4)
        for box in boxes:
            x, y, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('YOLO V8 DETECTION', frame)

    frame_counter += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
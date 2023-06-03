import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('./dataset_yolov8_bb/runs/detect/train2/weights/best.pt')
cap = cv2.VideoCapture("./videos/video2.MP4")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./bb_detection_videos/video2BB.mp4', fourcc, 30.0, (640, 640))

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

    results = model.predict(source=frame, save=False, save_txt=False, conf=0.6)

    for result in results:
        # Detection
        boxes = result.boxes.xyxy
        for box in boxes:
            x, y, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('YOLO V8 DETECTION', frame)

    # Write the frame into the file 'video3.mp4'
    out.write(frame)

    frame_counter += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
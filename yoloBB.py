import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from ultralytics.yolo.utils.plotting import Annotator

RESOLUTION = 640

model = YOLO('./dataset_yolov8_bb/runs/detect/train2/weights/best.pt')
cap = cv2.VideoCapture("./videos/video2.MP4")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./bb_detection_videos/video2BB.mp4', fourcc, 30.0, (640, 640))

frame_counter = 0

# log
columns = ['frame', 'time', 'class', 'min_confidence', 'bounding_box_x', 'bounding_box_y', 'bounding_box_x2', 'bounding_box_y2']
df = pd.DataFrame(columns=columns)


# Loop through the frames of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    time = frame_counter / 60.0


    frame = cv2.resize(frame, (640, 640))

    results = model.predict(source=frame, save=False, save_txt=False, conf=0.6)

    '''for result in results:
        # Detection
        boxes = result.boxes.xyxy
        for box in boxes:
            x, y, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            #log
            df = df.append({'frame': frame_counter, 'time': 0,
                            'class': 'car', 'confidence': 0,
                            'bounding_box_x': x, 'bounding_box_y': y,
                            'bounding_box_x2': x2, 'bounding_box_y2': y2},
                            ignore_index=True)'''
    
    for r in results:
        
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
            #log
            x, y, x2, y2 = map(int, b)
            df = df.append({'frame': frame_counter, 'time': time,
                            'class': model.names[int(c)], 'min_confidence': 0.6,
                            'bounding_box_x': x/RESOLUTION, 'bounding_box_y': y/RESOLUTION,
                            'bounding_box_x2': x2/RESOLUTION, 'bounding_box_y2': y2/RESOLUTION},
                            ignore_index=True)
          
    frame = annotator.result()  

    cv2.imshow('YOLO V8 DETECTION', frame)

    # Write the frame into the file 'video3.mp4'
    out.write(frame)

    

    frame_counter += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

df.to_csv('./logs_bb/video2BB.csv', index=True)
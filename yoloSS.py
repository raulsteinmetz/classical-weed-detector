import cv2
from ultralytics import YOLO

model = YOLO('./dataset_yolo_v8_ss/runs/segment/train/weights/best.pt')
cap = cv2.VideoCapture("./videos/video1.MP4")

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

    results = model.predict(source=frame, save=False, save_txt=True, conf=0.05, show=True, boxes=False)
    result = results[0]
    masks = result.masks
    print(len(masks))

    frame_counter += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
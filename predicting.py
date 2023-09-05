import cv2
from ultralytics import YOLO

model = YOLO('./dataset_yolo_v8_ss/runs/segment/train/weights/best.pt')

image = cv2.imread('./images_to_predict/video1_1300.jpg')
image = cv2.resize(image, (640, 640))

results = model.predict(source=image, save=True, save_txt=False, conf=0.1, show=True, boxes=True)

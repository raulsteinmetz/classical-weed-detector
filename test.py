from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('./dataset_yolov8/runs/detect/train8/weights/best.pt')

# from ndarray
im = cv2.imread("/home/raul/Documents/workspace/projects/soyImageProcessing/images/sample2/sample2_645.jpg")
results = model.predict(source=im, save=False, save_txt=False)  # save predictions as labels

for result in results:
    print(results)
    print('AAAA')
    # Detection
    boxes = result.boxes.xyxy  # box with xywh format, (N, 4)
    for box in boxes:
        print('AAAA')
        x, y, x2, y2 = map(int, box)
        cv2.rectangle(im, (x, y), (x2, y2), (0, 255, 0), 2)

cv2.imshow('image', im)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
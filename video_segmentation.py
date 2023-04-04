import cv2
import segmentation as seg
import numpy as np

cap = cv2.VideoCapture("./videos/sample_video1.mp4")

# Loop through the frames of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # reducing resolution of the frame
    frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    green = seg.green_seg(frame)

    
    # Concatenate the original and segmented images side by side
    output_frame = np.concatenate((frame, green), axis=1)

    cv2.imshow("Video", output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
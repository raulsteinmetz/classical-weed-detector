import cv2
import segmentation as seg
import numpy as np
from pre_processing import gaussian

cap = cv2.VideoCapture("./videos/sample_video3.mp4")

# Create a named window for the contours
cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)

# Loop through the frames of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = gaussian(frame)
    mask = seg.final_mask(frame)

    frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    # Manipulating mask
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # gray scale
    mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on a black image
    contour_img = np.zeros_like(mask)
    contour_img = cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)

    # Show the original image and the mask with contours in different windows
    cv2.imshow("Original", frame)
    cv2.imshow("Binary", mask)
    cv2.imshow("Contours", contour_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

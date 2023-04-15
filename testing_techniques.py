import cv2
import segmentation as seg
import numpy as np
from pre_processing import gaussian

cap = cv2.VideoCapture("./videos/sample_video2.mp4")

scale = 0.2


# Loop through the frames of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = gaussian(frame)
    mask = seg.final_mask(frame)

    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Manipulating mask
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # gray scale
    _, mask_otsu = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu thresholding

    mask_adaptive = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    

    # Find edges in the mask
    edges = cv2.Canny(mask_otsu, 200, 400)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on a black image
    contour_img = np.zeros_like(mask_otsu)
    contour_img = cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)


    displayed = np.concatenate((mask_otsu, mask_adaptive), axis=0)
    displayed_aux = np.concatenate((contour_img, edges), axis=0)
    displayed = np.concatenate((displayed, displayed_aux), axis=1)    

    print(displayed.shape[1], displayed.shape[0])

    # Show the original image and the mask with edges and contours in different windows
    cv2.imshow("Original", frame)

    cv2.imshow("Binaries", displayed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

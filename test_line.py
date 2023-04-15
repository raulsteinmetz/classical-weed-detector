import numpy as np
import cv2
import segmentation as seg
from pre_processing import gaussian


img = cv2.imread('./images/sample_1/frame_1300.jpg')

img = cv2.resize(img, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_CUBIC)
cv2.imshow('raw image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = gaussian(img)
combined_mask = seg.final_mask(img)
cv2.imshow('segmented image', combined_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


combined_mask_gray = cv2.cvtColor(combined_mask, cv2.COLOR_BGR2GRAY)
cv2.imshow('segmented gray image', combined_mask_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret, binary_img = cv2.threshold(combined_mask_gray, 0, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary Image', binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perform connected component analysis to segment the image
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

# Loop over the centroids and print the coordinates of each center
for i in range(1, num_labels):
    # Extract the centroid coordinates
    cx, cy = centroids[i]

    # Print the centroid coordinates
    print(f"Cluster {i} center: ({int(cx)}, {int(cy)})")

import cv2
import ClassicalWeedDetector.segmentation as seg
import numpy as np
from ClassicalWeedDetector.preprocessing import gaussian


scale = 0.5

image = cv2.imread('./images/soy_weed.png')

image = gaussian(image)
mask = seg.final_mask(image)

image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

# Manipulating mask
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # gray scale
_, mask_otsu = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu thresholding


# Find edges in the mask
# blur
# mask_otsu = cv2.GaussianBlur(mask_otsu, (7, 7), 0)
# opening
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
opened = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel)


edges = cv2.Canny(opened, 100, 200)

# Define the block size
block_size = 40
# Define the threshold for white pixels in the block
threshold = 180
# Iterate through the binary image to find blocks with more white pixels than the threshold
height, width = edges.shape
for i in range(0, height-block_size, block_size):
    for j in range(0, width-block_size, block_size):
        block = edges[i:i+block_size, j:j+block_size]
        if cv2.countNonZero(block) > threshold:
            print(f"Block found at ({i}, {j}) with {cv2.countNonZero(block)} white pixels")
            # Draw a red rectangle around the block
            cv2.rectangle(image, (j, i), (j+block_size, i+block_size), (0, 0, 255), 2)



cv2.imshow("Original", image)
cv2.imshow("Binary", edges)

cv2.waitKey()
cv2.destroyAllWindows()

import cv2
import numpy as np

# test image
img = cv2.imread('./images/sample_1/frame_1200.jpg')

# resize
img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

# Convert the image to HSV color space
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of green color in HSV
lower_green = np.array([36, 25, 25])
upper_green = np.array([86, 255, 255])

# Create a mask with the green pixels
green_mask = cv2.inRange(hsv_img, lower_green, upper_green)


# Apply morphological operations to remove noise and smooth the masks
kernel = np.ones((5,5),np.uint8)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

# Apply the masks to extract the green and brown areas
green_areas = cv2.bitwise_and(img, img, mask=green_mask)


# Concatenate the original and segmented images side by side
output_img = np.concatenate((img, green_areas), axis=1)

# Display the results
cv2.imshow('Original vs Green', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
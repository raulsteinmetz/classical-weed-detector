import cv2
import numpy as np

def green_seg(img):
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

    return green_areas
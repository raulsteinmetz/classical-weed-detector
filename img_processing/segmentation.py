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

    # Erode and dilate the green mask
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.erode(green_mask, kernel, iterations=1) # reduce noise
    # green_mask = cv2.dilate(green_mask, kernel, iterations=3) # soy more aparent

    # use this to create the line
    green_mask = cv2.dilate(green_mask, kernel, iterations=8)

    # Return the green mask
    return green_mask

def final_mask(img):
    green_mask = green_seg(img)
    return cv2.bitwise_and(img, img, mask=green_mask)



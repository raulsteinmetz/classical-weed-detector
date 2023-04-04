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

    # Return the green mask
    return green_mask

def brown_seg(img):
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds of dark brown color in HSV
    lower_brown = np.array([10, 60, 20])
    upper_brown = np.array([20, 255, 90])
    # Create a mask for dark brown pixels
    brown_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    # Apply morphological operations to remove noise and smooth the masks
    kernel = np.ones((5,5),np.uint8)
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel)

    # Return the brown mask
    return brown_mask

def final_mask(img):
    # Apply green segmentation
    green_mask = green_seg(img)

    # Apply brown segmentation
    brown_mask = brown_seg(img)

    # Combine the two masks by bitwise ORing them together
    combined_mask = cv2.bitwise_or(green_mask, brown_mask)

    # Apply the combined mask to the original image to extract both green and brown areas
    return cv2.bitwise_and(img, img, mask=combined_mask)

import cv2
import numpy as np

def gaussian(img, ksize=(5, 5)):
    # Apply Gaussian filter
    sigmaX = 0      # Standard deviation in X direction (0 means it is computed from ksize)
    sigmaY = 0      # Standard deviation in Y direction (0 means it is computed from ksize)
    return cv2.GaussianBlur(img, ksize, sigmaX, sigmaY)

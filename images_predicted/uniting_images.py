import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the three images
image1 = cv2.imread('video5_1500.jpg')
image2 = cv2.imread('video5_1500P.jpg')
image3 = cv2.imread('video5_1500PL.jpg')


# Define the spacing between images
spacing = 20  # Adjust as needed

# Create white space
white_space = np.ones((max(image1.shape[0], image2.shape[0], image3.shape[0]), spacing, 3), dtype=np.uint8) * 255

# Combine images horizontally with white space
combined_image = np.hstack((image1, white_space, image2, white_space, image3))

# Save the combined image
cv2.imwrite('image6.jpg', combined_image)

# Display the combined image
cv2.imshow('Combined Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

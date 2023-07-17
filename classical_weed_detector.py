import cv2
import ClassicalWeedDetector.segmentation as seg
import numpy as np
from ClassicalWeedDetector.preprocessing import gaussian

cap = cv2.VideoCapture("./videos/video2.MP4")

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
    threshold = 200
    
    # Define a list to keep track of the blocks with more white pixels than the threshold
    blocks = []

    # Iterate through the binary image to find blocks with more white pixels than the threshold
    height, width = edges.shape
    for i in range(0, height-block_size, block_size):
        for j in range(0, width-block_size, block_size):
            block = edges[i:i+block_size, j:j+block_size]
            if cv2.countNonZero(block) > threshold:
                # Check if the block has at least one neighbor that also has more white pixels than the threshold
                has_neighbor = False
                for x in range(max(i-block_size, 0), min(i+2*block_size, height-block_size), block_size):
                    for y in range(max(j-block_size, 0), min(j+2*block_size, width-block_size), block_size):
                        if x == i and y == j:
                            continue
                        neighbor = edges[x:x+block_size, y:y+block_size]
                        if cv2.countNonZero(neighbor) > threshold:
                            has_neighbor = True
                            break
                    if has_neighbor:
                        break
                
                # If the block has at least one neighbor, add it to the list of blocks to plot
                if has_neighbor:
                    blocks.append((i, j))
                    print(f"Block found at ({i}, {j}) with {cv2.countNonZero(block)} white pixels")

    # Plot the red rectangles only for the blocks with at least one neighbor
    for block in blocks:
        i, j = block
        cv2.rectangle(frame, (j, i), (j+block_size, i+block_size), (0, 0, 255), 2)



    # Show the original image and the mask with edges and contours in different windows
    cv2.imshow("Original", frame)

    cv2.imshow("Binaries", edges)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

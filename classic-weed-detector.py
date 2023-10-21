import cv2
import img_processing.segmentation as seg
import numpy as np
from img_processing.preprocessing import gaussian

cap = cv2.VideoCapture("./normal_res/20230114-GX010238.MP4")

# Define the output video file and codec
output_file = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

scale = 0.2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = gaussian(frame)
    mask = seg.final_mask(frame)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_otsu = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(opened, 100, 200)

    block_size = 125
    threshold = 950
    blocks = []

    height, width = edges.shape
    for i in range(0, height - block_size, block_size):
        for j in range(0, width - block_size, block_size):
            block = edges[i:i+block_size, j:j+block_size]
            if cv2.countNonZero(block) > threshold:
                has_neighbor = False
                for x in range(max(i - block_size, 0), min(i + 2*block_size, height - block_size), block_size):
                    for y in range(max(j - block_size, 0), min(j + 2*block_size, width - block_size), block_size):
                        if x == i and y == j:
                            continue
                        neighbor = edges[x:x+block_size, y:y+block_size]
                        if cv2.countNonZero(neighbor) > threshold:
                            has_neighbor = True
                            break
                    if has_neighbor:
                        break
                if has_neighbor:
                    blocks.append((i, j))
                    print(f"Block found at ({i}, {j}) with {cv2.countNonZero(block)} white pixels")

    for block in blocks:
        i, j = block
        cv2.rectangle(frame, (j, i), (j+block_size, i+block_size), (0, 0, 255), 2)

    if out is None:
        # Initialize the VideoWriter when the first frame is processed
        frame_height, frame_width = frame.shape[:2]
        out = cv2.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))

    out.write(frame)

    cv2.imshow("Original", frame)
    cv2.imshow("Binaries", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# Release the VideoWriter and close all OpenCV windows
if out is not None:
    out.release()

cv2.destroyAllWindows()

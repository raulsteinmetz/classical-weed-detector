import cv2
import segmentation as seg
import numpy as np
from pre_processing import gaussian

cap = cv2.VideoCapture("./videos/sample_video1.mp4")

# Get the video dimensions
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.25)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.25)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width*2, height))

# Loop through the frames of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # reducing resolution of the frame
    frame = gaussian(frame)

    # green and brown mask
    combined_mask = seg.final_mask(frame)

    # Concatenate the original and segmented images side by side
    output_frame = np.concatenate((frame, combined_mask), axis=1)

    # reduce the resolution to fit in a 1920x1080 monitor
    output_frame = cv2.resize(output_frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

    # Write the frame to the output video file
    out.write(output_frame)

    cv2.imshow("Video", output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
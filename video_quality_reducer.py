import cv2

# Load the input video
cap = cv2.VideoCapture('./processed_videos/sample_video1.mp4')

# Get the video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the output video dimensions and frame rate
out_width = 640
out_height = 480
out_fps = 30

# Create a VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('quality_reduced.mp4', fourcc, out_fps, (out_width, out_height))

# Loop through the frames of the input video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reduce the resolution of the frame
    frame = cv2.resize(frame, (out_width, out_height))

    # Write the frame to the output video
    out.write(frame)

    # Show the reduced quality video
    cv2.imshow('Reduced Quality Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()

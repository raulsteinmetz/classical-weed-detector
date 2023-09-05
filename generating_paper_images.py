import os
import cv2

cap = cv2.VideoCapture("./videos/video5.MP4")

i = 0
save_interval = 200  # Set the interval to save images every 500 frames
output_dir = './paper_images'

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Loop through the frames of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    i += 1

    frame = cv2.resize(frame, (640, 640))

    if i % save_interval == 0:
        save_path = os.path.join(output_dir, f"video5.jpg")
        cv2.imwrite(save_path, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
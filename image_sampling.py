import cv2

SAMPLE_JUMP = 100

# video path
cap = cv2.VideoCapture('./videos/sample_video1.mp4')

# output directory path
output_dir = './images/sample_1/'

# frame counter
frame_num = 0

# loop through the video frames
while True:
    # read frame
    ret, frame = cap.read()

    # breaks the loop if there arent any frames left
    if not ret:
        break

    # saving frame
    if frame_num % SAMPLE_JUMP == 0:
        cv2.imwrite(output_dir + f'frame_{frame_num:04d}.jpg', frame)

    # frame counter
    frame_num += 1


# release video capture object
cap.release()

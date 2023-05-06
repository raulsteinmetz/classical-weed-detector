import cv2
import os
import numpy as np

def sample(video_path, samples_path, name, frame_interval):
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    frame_count = 0
    sample_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame_count += 1

            size = 640
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


            if frame_count % frame_interval == 0:
                # Add Gaussian noise
                mean = 0
                stddev = 50
                noise = np.zeros(frame.shape, dtype=np.int16)
                cv2.randn(noise, mean, stddev)
                noised = cv2.add(frame, noise, dtype=cv2.CV_8UC3)

                # Flip the frame horizontally and vertically
                flipped_frame_horizontal = cv2.flip(noised, 1)
                flipped_frame_vertical = cv2.flip(noised, 0)

                # Save the original and flipped frames as PNG files
                sample_path = os.path.join(samples_path, f"{name}_{frame_count}.png")
                cv2.imwrite(sample_path, frame)
                flipped_horizontal_sample_path = os.path.join(samples_path, f"{name}_{frame_count}_flipped_horizontal.png")
                cv2.imwrite(flipped_horizontal_sample_path, flipped_frame_horizontal)
                flipped_vertical_sample_path = os.path.join(samples_path, f"{name}_{frame_count}_flipped_vertical.png")
                cv2.imwrite(flipped_vertical_sample_path, flipped_frame_vertical)

                sample_count += 1

            # Display the frame (optional)
            # cv2.imshow('Video', frame)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Sampled {sample_count} frames at a {frame_interval}-frame interval.")


def main():
    for i in np.arange(3):
        sample(f'./videos/sample_video{i + 1}.mp4',
            f'./images/sample{i + 1}/',
            f'sample{i + 1}',
            30)


if __name__ == '__main__':
    main()

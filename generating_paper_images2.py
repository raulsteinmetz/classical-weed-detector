import os
from PIL import Image

import os
from PIL import Image, ImageOps

def combine_images(input_folder, output_file, border_size=5, border_color=(255, 255, 255)):
    # Create a list to hold the image objects
    images = []

    # Get all files in the input folder
    files = os.listdir(input_folder)

    # Sort the files if necessary (for numerical order, assuming filenames are like "image1.png", "image2.png", etc.)
    files.sort()

    # Filter only PNG files (you can add other formats if needed)
    png_files = [file for file in files if file.lower().endswith(".jpg")]

    # Load the images from the folder and add them to the list
    for file in png_files:
        image_path = os.path.join(input_folder, file)
        image = Image.open(image_path)
        images.append(image)

    # Determine the size of the final combined image
    width, height = images[0].size
    final_width = width * 3 + border_size * 2
    final_height = height * 2 + border_size * 2

    # Create a new blank image with the final size and white background
    combined_image = Image.new('RGB', (final_width, final_height), (255, 255, 255))

    # Paste each image onto the combined image in a 3x2 grid with white borders
    for row in range(2):
        for col in range(3):
            index = row * 3 + col
            if index < len(images):
                offset_x = col * (width + border_size * 2) + border_size
                offset_y = row * (height + border_size * 2) + border_size
                bordered_image = ImageOps.expand(images[index], border=border_size, fill=border_color)
                combined_image.paste(bordered_image, (offset_x, offset_y))

    # Save the final combined image
    combined_image.save(output_file)


if __name__ == "__main__":
    input_folder = './paper_images_augmentation/'  # Update this with the actual path to your input folder
    output_file = "augmentation.png"  # Update this with the desired output file name
    combine_images(input_folder, output_file)

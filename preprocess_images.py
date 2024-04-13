import os
import cv2

# Define the paths for the original and binary image folders
original_folder = 'images/original'
binary_folder = 'images/binary'

# Create the binary image folder if it doesn't exist
if not os.path.exists(binary_folder):
    os.makedirs(binary_folder)

# Get the list of image files in the original folder
image_files = os.listdir(original_folder)

# Iterate over each image file
for image_file in image_files:
    # Check if the file is an image
    if image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Construct the paths for the original and binary images
        original_path = os.path.join(original_folder, image_file)
        binary_path = os.path.join(binary_folder, image_file)

        # Load the original image
        original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

        # Perform preprocessing steps to convert the image to binary

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)

        # Apply adaptive thresholding to obtain a binary image
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # Save the binary image
        cv2.imwrite(binary_path, binary_image)

        print(f"Preprocessed and saved binary image: {binary_path}")
    else:
        print(f"Skipping non-image file: {image_file}")
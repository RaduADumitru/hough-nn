import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

# Define the paths for the original and binary image folders
original_folder = 'images/original'
binary_folder = 'images/binary'
labeled_folder = 'images/labeled'

# Create the binary image folder if it doesn't exist
if not os.path.exists(binary_folder):
    os.makedirs(binary_folder)

if not os.path.exists(labeled_folder):
    os.makedirs(labeled_folder)

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

        # preprocessing for nearest-neighbors

        # Use connectedComponentsWithStats to label each object
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

        # Create an empty image to store the labels
        labelled_image = np.zeros_like(original_image)

        # Iterate over the labels and assign each pixel inside an object its corresponding label
        for i in range(1, num_labels):
            # Check if the label corresponds to a white pixel
            if np.mean(original_image[labels == i]) < 255:
                labelled_image[labels == i] = i

        # Save the labelled image
        labeled_path = os.path.join(labeled_folder, image_file)
        cv2.imwrite(labeled_path, labelled_image)

        colors = [np.random.rand(3,) for _ in range(num_labels)]
        cmap = plt.cm.colors.ListedColormap(colors)

        # Display the labeled image
        plt.figure(figsize=(10,10))
        plt.imshow(labelled_image, cmap=cmap)
        plt.colorbar(ticks=range(num_labels), label='label')
        plt.show()


    else:
        print(f"Skipping non-image file: {image_file}")
import sample_functions
import cv2
import os
import plot
import hough_transform
import nearest_neighbors

if __name__ == "__main__":
    # load images
    # specify the directory containing the images
    hough_image_directory = 'images/binary'

    # get the list of image file names in the directory
    hough_image_files = os.listdir(hough_image_directory)

    # load each image
    for image_file in hough_image_files:
        image_path = os.path.join(hough_image_directory, image_file)
        if not image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            continue
        image = cv2.imread(image_path)

        # process the image
        # TODO: replace function call with hough_transform.hough_transform
        plot.plot_function(sample_functions.sample_function, image, image_file, 1, 8)
    
    nearest_neighbors_image_directory = 'images/labeled'
    nearest_neighbors_image_files = os.listdir(nearest_neighbors_image_directory)
    for image_file in nearest_neighbors_image_files:
        image_path = os.path.join(nearest_neighbors_image_directory, image_file)
        if not image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            continue
        image = cv2.imread(image_path)

        # process the image
        # TODO: same as above but for nearest-neighbors
    
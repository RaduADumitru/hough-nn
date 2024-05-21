from mpi4py import MPI
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


def hough_transform_partial(image, theta, rho_range, rho_res):
    """
    Apply the Hough Transform for a specific angle theta on a part of the image.
    """
    width, height = image.shape
    diag_len = int(np.sqrt(width**2 + height**2))
    num_rhos = int((2 * diag_len) / rho_res)
    start_rho, end_rho = rho_range

    accumulator = np.zeros((end_rho - start_rho,), dtype=np.int32)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    for y in range(height):
        for x in range(width):
            if image[x, y]:  # if the pixel is an edge
                rho = int(round(x * cos_theta + y * sin_theta) /
                          rho_res + num_rhos / 2)
                if start_rho <= rho < end_rho:
                    accumulator[rho - start_rho] += 1

    return accumulator


# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load image
if rank == 0:
    # Root process loads the image
    image = cv2.imread('binary_image.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to read image file.")
    image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
else:
    image = None

# Broadcast image to all processes
image = comm.bcast(image, root=0)

# Define the angle for which we are computing the Hough transform
theta = np.deg2rad(45)  # Example angle of 45 degrees

# Define Hough transform parameters
rho_res = 1
width, height = image.shape
diag_len = int(np.sqrt(width**2 + height**2))
num_rhos = int((2 * diag_len) / rho_res)

# Split work among processes
rho_chunk = num_rhos // size
start_rho = rank * rho_chunk
end_rho = start_rho + rho_chunk if rank != size - 1 else num_rhos
rho_range = (start_rho, end_rho)

# Start timing
start_time = time.time()

# Each process computes its part of the Hough transform
partial_accumulator = hough_transform_partial(image, theta, rho_range, rho_res)

# Gather results from all processes
total_accumulator = None
if rank == 0:
    total_accumulator = np.zeros((num_rhos,), dtype=np.int32)

comm.Gather(partial_accumulator, total_accumulator, root=0)

# Stop timing
end_time = time.time()

# Root process now has the total Hough transform accumulator
if rank == 0:
    print(total_accumulator)
    plt.plot(total_accumulator)
    plt.title('Hough Transform Accumulator')
    plt.xlabel('Rho')
    plt.ylabel('Votes')
    plt.savefig('hough_transform_accumulator.png')

    print(f"Time taken: {end_time - start_time:.4f} seconds")

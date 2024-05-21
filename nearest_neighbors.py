from mpi4py import MPI
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


def get_bands(image, theta, rho_range, rho_res):
    """
    Gather pixel values (labels) for each band in a specific part of the image. 
    """
    width, height = image.shape
    diag_len = int(np.sqrt(width**2 + height**2))
    num_rhos = int((2 * diag_len) / rho_res)
    start_rho, end_rho = rho_range
   
    bands = [[] for _ in range(end_rho - start_rho)]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    theta_deg = np.rad2deg(theta) % 360

    y_range = range(height)
    if 0 <= theta_deg <= 90 or 180 <= theta_deg <= 270:
        y_range = range(height-1, -1, -1)
        
    for y in y_range:
        for x in range(width):
            rho = int(round(x * cos_theta + y * sin_theta) /
                        rho_res + num_rhos / 2)
            if start_rho <= rho < end_rho:
                bands[rho - start_rho].append(image[x, y])

    return bands


def compute_image(width, height, bands: list, theta, rho_range):
    diag_len = int(np.sqrt(width**2 + height**2))
    num_rhos = int((2 * diag_len) / rho_res)
    start_rho, end_rho = rho_range

    image = np.empty((width, height))

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    theta_deg = np.rad2deg(theta) % 360

    y_range = range(height)
    if 0 <= theta_deg <= 90 or 180 <= theta_deg <= 270:
        y_range = range(height-1, -1, -1)
        
    for y in y_range:
        for x in range(width):
            rho = int(round(x * cos_theta + y * sin_theta) /
                        rho_res + num_rhos / 2)
            if start_rho <= rho < end_rho:
                pixel = bands[rho - start_rho].pop(0)

                if pixel == -1:
                    image[x, y] = 255
                else:
                    image[x, y] = pixel

    return image


def get_nn(bands):
    """
    Compute neighbors and the distance to them for each pixel in each band. 
    """
    nn_bands, nn_dist_bands = [], []

    for band in bands:
        nn_band, nn_dist_band = [], []

        if len(band) == 1:
            nn_band.append(-1)
            nn_dist_band.append(-1)
        elif len(band) != 0:
            for i, p in enumerate(band):
                if p == 0:
                    nn_band.append(-1)
                    nn_dist_band.append(-1)
                    continue

                left_n, right_n = None, None

                for j in range(i-1, -1, -1):
                    if band[j] != p and band[j] != 0:
                        left_n = j
                        continue

                for j in range(i+1, len(band)):
                    if band[j] != p and band[j] != 0:
                        right_n = j
                        continue

                n = None
                if left_n == right_n == None:
                    nn_band.append(-1)
                    nn_dist_band.append(-1) 
                elif left_n == None:
                    n = right_n
                    nn_band.append(band[n])
                    nn_dist_band.append(right_n - i)
                elif right_n == None:
                    n = left_n
                    nn_band.append(band[n])
                    nn_dist_band.append(i - left_n)
                else:
                    left_dist = i - left_n
                    right_dist = right_n - i
                    min_dist = min(left_dist, right_dist)

                    if min_dist == left_dist:
                        nn_band.append(band[left_n])
                    else:
                        nn_band.append(band[right_n])

                    nn_dist_band.append(min_dist)

        nn_bands.append(nn_band)
        nn_dist_bands.append(nn_dist_band)

    return nn_bands, nn_dist_bands


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    image = cv2.imread('images/labeled/fruits_vector.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to read image file.")
else:
    image = None

image = comm.bcast(image, root=0)

rho_res = 1
width, height = image.shape
diag_len = int(np.sqrt(width**2 + height**2))
num_rhos = int((2 * diag_len) / rho_res)

rho_chunk = num_rhos // size
start_rho = rank * rho_chunk
end_rho = start_rho + rho_chunk if rank != size - 1 else num_rhos
rho_range = (start_rho, end_rho)

theta = np.deg2rad(45)

start_time = time.time()

partial_bands = get_bands(image, theta, rho_range, rho_res)
partial_nn, partial_nn_dist = get_nn(partial_bands)

total_nn = comm.gather(partial_nn, root=0)
total_nn_dist = comm.gather(partial_nn_dist, root=0)

end_time = time.time()

if rank == 0:
    concat = []
    for list in total_nn:
        concat += list
    total_nn = concat

    concat = []
    for list in total_nn_dist:
        concat += list
    total_nn_dist = concat

    print(f"Time taken: {end_time - start_time:.4f} seconds")

    image = compute_image(width, height, total_nn, theta, (0, num_rhos))
    plt.imsave('images/nn/hough_nn-{}.png'.format(int(np.rad2deg(theta))), image, cmap='gray')
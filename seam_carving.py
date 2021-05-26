import time
from math import sqrt

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, types
from numba.extending import overload
from scipy.ndimage import generic_filter
from skimage.io import imshow
from poisson_solver import poisson_solver


# This allows numba to build a numpy array from another numpy array
# collected from: https://github.com/numba/numba/issues/4470#issuecomment-523395410
@overload(np.array)
def np_array_ol(x):
    if isinstance(x, types.Array):
        def impl(x):
            return np.copy(x)

        return impl


def sobel_derivative(grid):
    """
    :param grid: A 3 x 3 grid
    :return: Gradient of the grid, calculated in x and y direction
    numba compilation is only useful if the array contains many (>1000) elements, so we won't @njit here
    """
    sobel, grid = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), grid.reshape((3, 3))
    return sqrt(np.sum(sobel * grid) ** 2 + np.sum(sobel.T * grid) ** 2)


def e1_gray(image):
    """
    :param image: a grayscale image as 2D numpy array
    :return: a 2D numpy array with the same size as 'image', this array holds e1 energy for every pixel in the image.
    This energy represents the sum of derivative in x and y direction for each pixel, calculated via the sobel kernel.
    """
    return generic_filter(image, sobel_derivative, size=(3, 3))


@njit
def e1_gray_numba(image):
    """
    :param image: a grayscale image as 2D numpy array
    :return: a 2D numpy array with the same size as 'image', this array holds e1 energy for every pixel in the image.
    This energy represents the sum of derivative in x and y direction for each pixel, calculated via the sobel kernel.
    This implementation is faster than the 'e1_gray(image)' function.
    """
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    rows, cols = image.shape
    result = np.zeros((rows, cols))
    # numpy.pad is not supported by numba, so we'll have to do this manually
    pad_img = np.zeros((rows + 2, cols + 2))
    # corners
    pad_img[0, 0], pad_img[0, -1], pad_img[-1, 0], pad_img[-1, -1] = image[0, 0], image[0, -1], image[-1, 0], image[
        -1, -1]
    # sides
    pad_img[0, 1:-1], pad_img[1:-1, 0], pad_img[-1, 1:-1], pad_img[1:-1, -1] = image[0], image[:, 0], image[-1], image[
                                                                                                                 :, -1]
    # bulk of the data
    pad_img[1:-1, 1:-1] = image
    for row_idx in range(rows):
        for col_idx in range(cols):
            grid = pad_img[row_idx:row_idx + 3, col_idx:col_idx + 3]
            result[row_idx, col_idx] += sqrt(np.sum(sobel * grid) ** 2 + np.sum(sobel.T * grid) ** 2)
    return result


def e1_colour(image):
    """
    :param image: a color image as 3D numpy array
    :return: a 2D numpy array holding the e1 energy for every pixel in the image.
    """
    return e1_gray(image[:, :, 0]) / 3 + e1_gray(image[:, :, 1]) / 3 + e1_gray(image[:, :, 2]) / 3


@njit
def e1_colour_numba(image):
    """
    :param image: a colour image as 3D numpy array
    :return: a 2D numpy array holding the e1 energy for every pixel in the image.
    """
    rows, cols, colour = image.shape
    result = np.zeros((rows, cols))
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel3D = np.zeros((3, 3, colour))
    sobel3DT = np.zeros((3, 3, colour))
    for channel in range(colour):
        sobel3D[channel] = sobel
        sobel3DT[channel] = sobel.T
    # numpy.pad is not supported by numba, so we'll have to do this manually
    pad_img = np.zeros((rows + 2, cols + 2, 3))
    # corners
    pad_img[0, 0], pad_img[0, -1], pad_img[-1, 0], pad_img[-1, -1] = image[0, 0], image[0, -1], image[-1, 0], image[
        -1, -1]
    # sides
    pad_img[0, 1:-1], pad_img[1:-1, 0], pad_img[-1, 1:-1], pad_img[1:-1, -1] = image[0], image[:, 0], image[-1], image[
                                                                                                                 :, -1]
    # bulk of the data
    pad_img[1:-1, 1:-1] = image
    for row_idx in range(rows):
        for col_idx in range(cols):
            grid = pad_img[row_idx:row_idx + 3, col_idx:col_idx + 3]
            result[row_idx, col_idx] += sqrt(np.sum(sobel3D * grid.T) ** 2 + np.sum(sobel3DT * grid.T) ** 2) / 3
    return result.astype(np.float32)


def entropy_helper(grid):
    """
    Helper function for the entropy calculation
    :param grid: an nxm grid
    :return: the Shannon entropy of the grid
    """
    grid = grid.astype(np.int64)
    probabilities = np.bincount(grid) / grid.size
    probabilities = probabilities[probabilities != 0]
    return -(probabilities * np.log2(probabilities)).sum()


def entropy(image):
    """
    :param image: 2D image as numpy array
    :return: entropy of the 2D image for each pixel.
    The entropy is calculated with a 9x9 grid, as indicated in Avidan & Shamir (2007).
    """
    return generic_filter(np.array(image), entropy_helper, size=(9, 9), output=np.zeros(image.shape))


def entropy_gray(image):
    """
    :param image: 2D image as numpy array
    :return: e1 error + entropy of the 2D image for each pixel.
    """
    return e1_gray(image) + entropy(image)


@njit
def entropy_gray_numba(image):
    """
    :param image: 2D image as numpy array
    :return: 2D numpy array containing the e1 error + entropy of the image.
    The entropy is calculated with a 9x9 grid, as indicated in Avidan & Shamir (2007).
    """
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    rows, cols = image.shape
    result = np.zeros((rows, cols))
    # numpy.pad is not supported by numba, so we'll have to do this manually
    pad_img = np.zeros((rows + 2, cols + 2))
    # corners
    pad_img[0, 0], pad_img[0, -1], pad_img[-1, 0], pad_img[-1, -1] = image[0, 0], image[0, -1], image[-1, 0], image[
        -1, -1]
    # sides
    pad_img[0, 1:-1], pad_img[1:-1, 0], pad_img[-1, 1:-1], pad_img[1:-1, -1] = image[0], image[:, 0], image[-1], image[
                                                                                                                 :, -1]
    # bulk of the data
    pad_img[1:-1, 1:-1] = image
    for row_idx in range(rows):
        row_start, row_end = max(0, row_idx - 4), min(rows, row_idx + 5)
        for col_idx in range(cols):
            col_start, col_end = max(0, col_idx - 4), min(cols, col_idx + 5)
            grid = image[row_start:row_end, col_start:col_end].flatten().astype(np.uint8)
            probabilities = np.bincount(grid) / grid.size
            probabilities = probabilities[probabilities != 0]
            result[row_idx, col_idx] += -(probabilities * np.log2(probabilities)).sum()
            grid = pad_img[row_idx:row_idx + 3, col_idx:col_idx + 3]
            result[row_idx, col_idx] += sqrt(np.sum(sobel * grid) ** 2 + np.sum(sobel.T * grid) ** 2)
    return result


def entropy_colour(image):
    """
    :param a colour image as 3D numpy array
    :return: 2D numpy array containing the e1 error + entropy of the image.
    """
    return entropy(image[:, :, 0]) / 3 + entropy(image[:, :, 1]) / 3 + entropy(image[:, :, 2]) / 3


@njit
def entropy_colour_numba(image):
    """
    :param image: a colour image as 3D numpy array
    :return: 2D numpy array containing the e1 error + entropy of the image.
    """
    rows, cols, colour = image.shape
    result = np.zeros((rows, cols))
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel3D = np.zeros((3, 3, colour))
    sobel3DT = np.zeros((3, 3, colour))
    for channel in range(colour):
        sobel3D[channel] = sobel
        sobel3DT[channel] = sobel.T
    # numpy.pad is not supported by numba, so we'll have to do this manually
    pad_img = np.zeros((rows + 2, cols + 2, 3))
    # corners
    pad_img[0, 0], pad_img[0, -1], pad_img[-1, 0], pad_img[-1, -1] = image[0, 0], image[0, -1], image[-1, 0], image[
        -1, -1]
    # sides
    pad_img[0, 1:-1], pad_img[1:-1, 0], pad_img[-1, 1:-1], pad_img[1:-1, -1] = image[0], image[:, 0], image[-1], image[
                                                                                                                 :, -1]
    # bulk of the data
    pad_img[1:-1, 1:-1] = image
    for row_idx in range(rows):
        row_start, row_end = max(0, row_idx - 4), min(rows, row_idx + 5)
        for col_idx in range(cols):
            col_start, col_end = max(0, col_idx - 4), min(cols, col_idx + 5)
            # e1 energy
            grid = pad_img[row_idx:row_idx + 3, col_idx:col_idx + 3]
            result[row_idx, col_idx] += sqrt(np.sum(sobel3D * grid.T) ** 2 + np.sum(sobel3DT * grid.T) ** 2) / 3
            # entropy per R,G and B band
            for color in range(colour):
                grid = image[row_start:row_end, col_start:col_end, color].flatten().astype(np.uint8)
                probabilities = np.bincount(grid) / grid.size
                probabilities = probabilities[probabilities != 0]
                result[row_idx, col_idx] += -(probabilities * np.log2(probabilities)).sum()
    return result


@njit
def hog_colour_numba(image):
    """
    :param image: a colour image as 3D numpy array
    :return: 2D numpy array containing the e1 error / max(histogram_of_gradients) of the image.
    """
    rows, cols, colour = image.shape
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel3D = np.zeros((3, 3, 3))
    sobel3DT = np.zeros((3, 3, 3))
    sobel3D[0], sobel3D[1], sobel3D[2] = sobel, sobel, sobel
    sobel3DT[0], sobel3DT[1], sobel3DT[2] = sobel.T, sobel.T, sobel.T
    dx, dy = np.zeros((rows, cols)), np.zeros((rows, cols))
    # numpy.pad is not supported by numba, so we'll have to do this manually
    pad_img = np.zeros((rows + 2, cols + 2, 3))
    # corners
    pad_img[0, 0], pad_img[0, -1], pad_img[-1, 0], pad_img[-1, -1] = image[0, 0], image[0, -1], image[-1, 0], image[
        -1, -1]
    # sides
    pad_img[0, 1:-1], pad_img[1:-1, 0], pad_img[-1, 1:-1], pad_img[1:-1, -1] = image[0], image[:, 0], image[-1], image[
                                                                                                                 :, -1]
    # bulk of the data
    pad_img[1:-1, 1:-1] = image
    e1 = np.zeros((rows, cols))
    # First calculate the e1 energy matrix
    for row_idx in range(rows):
        for col_idx in range(cols):
            grid = pad_img[row_idx:row_idx + 3, col_idx:col_idx + 3]
            dx3, dy3 = sobel3D * grid.T, sobel3DT * grid.T
            # As indicated in Avidan & Shamir (2007), the highest of the three (R, G and B) values will be used,
            # for the calculation of the gradient, both in x and y direction.
            dx[row_idx, col_idx] = max(np.sum(dx3[:, :, 0]), np.sum(dx3[:, :, 1]), np.sum(dx3[:, :, 2]))
            dy[row_idx, col_idx] = max(np.sum(dy3[:, :, 0]), np.sum(dy3[:, :, 1]), np.sum(dy3[:, :, 2]))
            e1[row_idx, col_idx] = np.sqrt(np.sum(dx3) ** 2 + np.sum(dy3) ** 2)
    gradient = np.sqrt(np.square(dx) + np.square(dy))
    angle = (np.arctan2(dy, dx) * 360 / np.pi) % 180
    bin_num, bin_spread = (angle // 20).astype(np.uint8), angle % 20 * 0.05
    max_hist_of_grad = np.zeros((rows, cols))
    # Calculate histogram of gradients
    for row_idx, gradient_row in enumerate(gradient):
        row_end = min(row_idx + 11, rows)  # Make sure we don't index more than row length
        for col_idx, gradient_cell in enumerate(gradient_row):
            bins = np.zeros(9)
            col_end = min(col_idx + 11, cols)  # Make sure we don't index more than column length
            gradient_grid = gradient[row_idx:row_end, col_idx:col_end].flatten()
            bin_num_grid = bin_num[row_idx:row_end, col_idx:col_end].flatten()
            bin_spread_grid = bin_spread[row_idx:row_end, col_idx:col_end].flatten()
            for idx, bin in enumerate(bin_num_grid):
                bins[bin] += gradient_grid[idx] * (1 - bin_spread_grid[idx])
                bins[(bin + 1) % 9] += gradient_grid[idx] * bin_spread_grid[idx]
            max_hist_of_grad[row_idx, col_idx] += max(bins)
    return e1 / max_hist_of_grad


@njit
def find_vertical_seams(e_matrix, amount=1):
    """
    :param e_matrix: The energy representation of an image, should be 2D numpy array
    :param amount: The number of vertical seams to find
    :return:    (0) 2D numpy array, each ROW contains the COLUMN indices of a vertical seam path
                (1) the total energy of the minimal energy path of the last seam
    """
    e_matrix = e_matrix.astype(np.float32)
    rows, cols = e_matrix.shape
    seams = np.zeros((amount, rows))
    for num in range(amount):
        # path_e will store the energy of the minimum energy path available for each pixel,
        # path will store the column index of the pixel above it with the minimal energy path
        path_e, path = np.zeros((rows, cols), dtype=np.float32), np.zeros((rows, cols))
        path_e[0] = e_matrix[0]
        # Find the minimum energy path (top to bottom) for each pixel
        for row_idx in range(1, rows):
            for col_idx in range(cols):
                # The neighbours are the three pixels above it (border pixels only have two neighbours)
                neighbours = path_e[row_idx - 1, max(0, col_idx - 1):min(col_idx + 2, cols)]

                # If the pixel is already in a seam, then we don't include it anymore
                if np.isinf(e_matrix[row_idx, col_idx]):
                    path[row_idx, col_idx] = np.inf
                    path_e[row_idx, col_idx] = np.inf

                # If all the neighbours are already in seams then we won't be able to use this pixel anymore either
                elif np.all(np.isinf(neighbours)):
                    e_matrix[row_idx, col_idx] = np.inf
                    path[row_idx, col_idx] = np.inf
                    path_e[row_idx, col_idx] = np.inf

                # The pixel is not yet in a seam and still has 'non-seam' neighbours, let's evaluate it
                else:
                    min_nbr_idx = max(0, col_idx + neighbours.argmin() - 1)
                    path[row_idx, col_idx] = min_nbr_idx
                    path_e[row_idx, col_idx] = e_matrix[row_idx, col_idx] + path_e[row_idx - 1, min_nbr_idx]

        # Backtrack the minimal path from bottom to top
        seam_path = np.zeros(rows)  # Stores the column indices for pixels in the lowest energy path
        col_idx = path_e[-1].argmin()  # Find the pixel with lowest total energy path in the last row
        seam_path[-1] = col_idx
        # If we add this pixel to a seam we no longer want to be able to add it to another seam later on
        e_matrix[-1, col_idx] = np.inf
        for row_idx in range(rows - 1, 0, -1):  # Traverse from bottom of paths to top following minimal entropy path
            minimum_neighbour_col_idx = int(path[row_idx, col_idx])
            seam_path[row_idx - 1] = minimum_neighbour_col_idx
            # If we add this pixel to a seam we no longer want to be able to add it to another seam later on
            e_matrix[row_idx - 1, minimum_neighbour_col_idx] = np.inf
            col_idx = minimum_neighbour_col_idx
        seams[num] = seam_path
    return seams, np.min(path_e[-1])


@njit
def find_horizontal_seams(e_matrix, amount=1):
    """
    :param e_matrix: The energy of an image, should be a 2D numpy array
    :param amount: The number of vertical seams to find
    :return:    (0) 2D numpy array, each ROW contains the ROW indices of a horizontal seam path
                (1) the total energy of the minimal energy path of the last seam
    """
    return find_vertical_seams(e_matrix.T, amount)


def remove_column(image, energy_matrix):
    """
    :param image: RGB image as 3D numpy array
    :param energy_matrix: The energy of an image, should be a 2D numpy array
    :return:    (0) the reduced image
                (1) the values of the pixels that were reduced
                (2) the COLUMN indices of the pixels that were removed (seam)
    """
    seam = find_vertical_seams(energy_matrix)[0][0]
    values = image[np.arange(image.shape[1]) == np.array(seam).reshape(seam.shape[0], 1)].reshape(image.shape[0], 3)
    image = image[np.arange(image.shape[1]) != np.array(seam).reshape(seam.shape[0], 1)].reshape(image.shape[0], -1, 3)
    return image, values, seam


@njit
def remove_column_seam_numba(image, seam):
    """
    :param image: RGB image as 3D numpy array
    :param seam: The seam containing the ROW indices that should be removed, should be a 1D numpy array
    :return:    (0) the reduced image
                (1) the values of the pixels that were reduced
    """
    values = np.zeros((image.shape[0], 3))
    new_image = np.zeros((image.shape[0], image.shape[1] - 1, 3))
    for row_idx, col_idx in enumerate(seam):
        col_idx = int(col_idx)
        values[row_idx] = image[row_idx, col_idx]
        new_image[row_idx, :col_idx] = image[row_idx, :col_idx]
        new_image[row_idx, col_idx:] = image[row_idx, col_idx + 1:]
    return new_image.astype(np.float32), values


@njit
def remove_row_seam_numba(image, seam):
    """
    :param image: RGB image as 3D numpy array
    :param seam: The seam containing the COLUMN indices that should be removed, should be a 1D numpy array
    :return:    (0) the reduced image
                (1) the values of the pixels that were reduced
    """
    rows, cols, colours = image.shape
    image_T = np.zeros((cols, rows, colours))
    new_image = np.zeros((rows - 1, cols, colours))
    image_T[:, :, 0], image_T[:, :, 1], image_T[:, :, 2] = image[:, :, 0].T, image[:, :, 1].T, image[:, :, 2].T
    new_image_T, values = remove_column_seam_numba(image_T, seam)
    new_image[:, :, 0], new_image[:, :, 1], new_image[:, :, 2] = new_image_T[:, :, 0].T, new_image_T[:, :,
                                                                                         1].T, new_image_T[:, :, 2].T
    return new_image.astype(np.float32), values


def remove_mask(image, energy_function, mask_to_remove, mask_to_keep):
    mask_to_remove = mask_to_remove.astype(np.float32)
    mask_to_keep = mask_to_keep.astype(np.float32)
    # Find the row and column indices which contain at least one pixel to be removed
    remove_mask_rows, remove_mask_cols = np.where(mask_to_remove.any(axis=1))[0], np.where(mask_to_remove.any(axis=0))[0]
    remove_mask_height = np.max(remove_mask_rows) - np.min(remove_mask_rows)
    remove_mask_width = np.max(remove_mask_cols) - np.min(remove_mask_cols)
    if remove_mask_height >= remove_mask_width:
        seam_find_function = find_vertical_seams
        seam_remove_function = remove_column_seam_numba
    else:
        seam_find_function = find_horizontal_seams
        seam_remove_function = remove_row_seam_numba
    # Assign a high negative value to each pixel in the remove mask and high positive value to those in the keep mask
    max_size = np.max((mask_to_remove.shape[0], mask_to_remove.shape[1]))
    mask = np.where(mask_to_remove, -100*mask_to_remove*max_size, 0)
    mask += 100*mask_to_keep*max_size
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    amount_to_remove = np.sum(mask_to_remove)
    while np.min(mask_3d) < 0:
        e_matrix = energy_function(image)
        # By adding mask3d, the seams will preferentially include pixels in the mask to be removed (negative values)
        # and exclude pixels in the mask to be kept (high positive values)
        e_matrix += mask_3d[:, :, 0]
        seam = seam_find_function(e_matrix)[0][0]
        image = seam_remove_function(image, seam)[0]
        mask_3d = seam_remove_function(mask_3d, seam)[0]
        new_amount_to_remove = np.sum(np.where(mask_3d[:, :, 0] < 0, 1, 0))
        if amount_to_remove == new_amount_to_remove:
            # This happens when pixels to be removed are surrounded by pixels that cannot be removed
            # This can be because either the user drew impossible masks (e.g., a red zone encircled by a green zone)
            # Or this can also happen due to the seam carving process and is ~impossible to prevent
            # If this happens then we'll stop otherwise we'll keep removing rows/columns until the image is more or less
            # completely gone
            break
        amount_to_remove = new_amount_to_remove
    return image

def process_gradient_domain(function, image, new_seam):
    """
    :param function: function to perform operation on, can be remove_row_seam_numba or remove_column_seam_numba
    :param image: RGB image as 3D numpy array
    :param seam: The seam containing the ROW/COLUMN indices that should be removed, should be a 1D numpy array
    :return:    (0) the reduced image
                (1) the values of the pixels that were reduced
    """
    # Compute derivatives before seam is removed
    gx, gy = forward_derivative(image.astype(np.float32))
    
    image, new_values = function(image, new_seam)
    
    # Remove seam from derivatives
    gx, _ = function(gx, new_seam)
    gy, _ = function(gy, new_seam)
               
    # Reconstruct original image for all color channels
    image[:,:,0] = poisson_solver(gx[:,:,0], gy[:,:,0], image.astype(np.float32)[:,:,0])
    image[:,:,1] = poisson_solver(gx[:,:,1], gy[:,:,1], image.astype(np.float32)[:,:,1])
    image[:,:,2] = poisson_solver(gx[:,:,2], gy[:,:,2], image.astype(np.float32)[:,:,2])
    
    return image, new_values

# Passing functions to numba compilated code is quite hard and complicated, so we won't numba compile this function
def remove_rows_and_cols(image, energy_function, rows_to_remove=0, cols_to_remove=0, gradient_domain=0):
    """
    :param image: RGB image as 3D numpy array
    :param energy_function: An energy function (e1_colour_numba, entropy_colour_numba or hog_colour_numba) to apply
    :param rows_to_remove: amount of rows to be removed
    :param cols_to_remove: amount of columns to be removed
    :return:    (0) the reduced image
                (1) the values of the pixels that were reduced
                (2) the indices of the pixels that were removed (seam)
                (3) the order by which rows (True) or columns (False) were removed

                Items last in the lists are the most recent removals, items in the front were removed early on.
    """
    seam = []
    values = []
    order = []  # True = row, False = column

    while rows_to_remove > 0 and cols_to_remove > 0:
        energy_matrix = energy_function(image)
        column_seam = find_vertical_seams(energy_matrix)
        row_seam = find_horizontal_seams(energy_matrix)
        if row_seam[1] <= column_seam[1]:
            new_seam = row_seam[0][0]

            if not gradient_domain:
                image, new_values = remove_row_seam_numba(image, new_seam)
            else:
                image, new_values = process_gradient_domain(remove_row_seam_numba, image, new_seam)
                
            order.append(True)
            rows_to_remove -= 1
        else:
            new_seam = column_seam[0][0]

            if not gradient_domain:
                image, new_values = remove_column_seam_numba(image, new_seam)
            else:
                image, new_values = process_gradient_domain(remove_column_seam_numba, image, new_seam)
                
            order.append(False)
            cols_to_remove -= 1
        seam.append(new_seam)
        values.append(new_values)

    while rows_to_remove > 0:  # The columns have already been removed
        energy_matrix = energy_function(image)
        new_seam = find_horizontal_seams(energy_matrix)[0][0]

        if not gradient_domain:
            image, new_values = remove_row_seam_numba(image, new_seam)
        else:
            image, new_values = process_gradient_domain(remove_row_seam_numba, image, new_seam)
            
        order.append(True)
        seam.append(new_seam)
        values.append(new_values)
        rows_to_remove -= 1

    while cols_to_remove > 0:  # The rows have already been removed
        energy_matrix = energy_function(image)
        new_seam = find_vertical_seams(energy_matrix)[0][0]

        if not gradient_domain:
            image, new_values = remove_column_seam_numba(image, new_seam)
        else:
            image, new_values = process_gradient_domain(remove_column_seam_numba, image, new_seam)
            
        order.append(False)
        seam.append(new_seam)
        values.append(new_values)
        cols_to_remove -= 1

    return image.astype(np.uint8), values, seam, order


@njit
def add_column_seam_numba(image, seams):
    """
    :param image: RGB image as 3D numpy array
    :param seams: The seams where the addition should happen, 2D numpy array
    :return: the increased image
    """
    # We cannot simply start adding the seams as adding a column, will alter the indices to the right of that added
    # column. So in order to prevent this, we will traverse row by row and perform the additions/insertions right
    # to left. This allows us to insert the pixels at the places where they were intended. We'll start by transposing
    # 'new_seams' so that each row of this matrix now contains the column values of that row
    # where an insertion should happen.
    rows, cols, colours = image.shape
    columns_to_add = len(seams)
    new_image = np.zeros((rows, cols+columns_to_add, colours))
    seams = seams.T
    print(seams.shape, image.shape)
    for row_idx, row in enumerate(seams):
        new_row = np.zeros((cols + columns_to_add, colours))
        row = np.sort(row)
        previous_index = 0
        for col_num, col_idx in enumerate(row):
            col_idx = int(col_idx)
            new_row[previous_index + col_num:col_idx + col_num] = image[row_idx, previous_index:col_idx]
            new_row[col_idx + col_num] = image[row_idx, col_idx - 1] / 2 + image[row_idx, col_idx] / 2
            previous_index = col_idx
        new_row[previous_index + col_num+1::] = image[row_idx, previous_index::]
        new_image[row_idx] = new_row
    return new_image.astype(np.uint8)


@njit
def add_row_seam_numba(image, seams):
    """
    :param image: RGB image as 3D numpy array
    :param seams: The horizontal seams of the image, should be a 2D numpy array
    :return: the increased image
    """
    rows, cols, colours = image.shape
    image_T = np.zeros((cols, rows, colours))
    new_image = np.zeros((rows + 1, cols, colours))
    image_T[:, :, 0], image_T[:, :, 1], image_T[:, :, 2] = image[:, :, 0].T, image[:, :, 1].T, image[:, :, 2].T
    new_image_T = add_column_seam_numba(image_T, seams)
    new_image[:, :, 0], new_image[:, :, 1], new_image[:, :, 2] = new_image_T[:, :, 0].T, new_image_T[:, :,
                                                                                         1].T, new_image_T[:, :, 2].T
    return new_image


# Passing functions to numba compilated code is quite hard and complicated, so we won't numba compile this function
def add_columns(image, energy_function, columns_to_add=1):
    """
    :param image: RGB image as 3D numpy array
    :param energy_function: An energy function (e1_colour_numba, entropy_colour_numba or hog_colour_numba) to apply
    :param columns_to_add: amount of rows to be added
    :return:    (0) the increased image
                (1) the ROW indices of the pixels that were added (seam)

                Items last in the lists are the most recent additions, items in the front were added early on.
    """
    seams = []
    # The denominator can be tuned manually, if one wants to find too much seams at the same time
    # then it might be possible 'to run out of possible seams'. In theory, in a square image, three diagonal, adjacent
    # seams would already prevent any more seams to be found. You run into a sort of integer overflow when this happens
    # to you. So be wary of trying to add too much rows at a time! The higher the denominator, the less chance this has
    # of happening but also the higher the 'stretching artifacts' in the image will be. Perhaps we might include a check
    # to the 'find_vertical_seams' function to catch this.
    denominator = 4
    while columns_to_add > 0:
        if columns_to_add >= image.shape[0]//denominator:
            amount = image.shape[0]//denominator
        else:
            amount = columns_to_add
        energy_matrix = energy_function(image)
        new_seams = find_vertical_seams(energy_matrix, amount=amount)[0]
        image = add_column_seam_numba(image, new_seams)
        seams.extend(new_seams)
        columns_to_add -= amount

    return image.astype(np.uint8), seams


def add_rows(image, energy_function, rows_to_add=1):
    """
    :param image: RGB image as 3D numpy array
    :param energy_function: An energy function (e1_colour_numba, entropy_colour_numba or hog_colour_numba) to apply
    :param rows_to_add: amount of rows to be added
    :return:    (0) the increased image
                (1) the COLUMN indices of the pixels that were added (seam)

                Items last in the lists are the most recent additions, items in the front were added early on.
    """
    rows, cols, colours = image.shape
    image_T = np.zeros((cols, rows, colours))
    new_image = np.zeros((rows + rows_to_add, cols, colours))
    image_T[:, :, 0], image_T[:, :, 1], image_T[:, :, 2] = image[:, :, 0].T, image[:, :, 1].T, image[:, :, 2].T
    new_image_T, seams = add_columns(image_T, energy_function, rows_to_add)
    new_image[:, :, 0], new_image[:, :, 1], new_image[:, :, 2] = new_image_T[:, :, 0].T, new_image_T[:, :,
                                                                                         1].T, new_image_T[:, :, 2].T
    return new_image.astype(np.uint8), seams


@njit
def reconstruct_column_seam_numba(image, values, seam):
    """
    :param image: RGB image as 3D numpy array
    :param seam: The seam where the addition should happen, 1D numpy array
    :return:    (0) the increased image
                (1) the ROW indices of the pixels that were added (seam)
    """
    new_image = np.zeros((image.shape[0], image.shape[1] + 1, 3))
    for row_idx, col_idx in enumerate(seam):
        col_idx = int(col_idx)
        if not col_idx:  # If the index = 0
            col_idx += 1
        new_image[row_idx, :col_idx] = image[row_idx, :col_idx]
        new_image[row_idx, col_idx] = values[row_idx]
        new_image[row_idx, col_idx+1:] = image[row_idx, col_idx:]
    return new_image.astype(np.uint8)


@njit
def reconstruct_row_seam_numba(image, values, seam):
    """
    :param image: RGB image as 3D numpy array
    :param seam: The horizontal seam of the image, should be a 1D numpy array
    :return: the increased image
    """
    rows, cols, colours = image.shape
    image_T = np.zeros((cols, rows, colours))
    new_image = np.zeros((rows + 1, cols, colours))
    image_T[:, :, 0], image_T[:, :, 1], image_T[:, :, 2] = image[:, :, 0].T, image[:, :, 1].T, image[:, :, 2].T
    new_image_T = reconstruct_column_seam_numba(image_T, values, seam)
    new_image[:, :, 0], new_image[:, :, 1], new_image[:, :, 2] = new_image_T[:, :, 0].T, new_image_T[:, :,
                                                                                         1].T, new_image_T[:, :, 2].T
    return new_image

@njit
def forward_derivative(image):
    """
    :param image: a colour/grayscale image as 2D numpy array as float32
    :return:    (0) 2D numpy array, forward derivative for x
                (1) 2D numpy array, forward derivative for y
    """

    resultx = np.zeros(image.shape)
    resulty = np.zeros(image.shape)
          
    resultx[:-1,:-1] = image[:-1,1:] - image[:-1,:-1]
    resulty[:-1,:-1] = image[1:,:-1] - image[:-1,:-1]
    
    return resultx, resulty

if __name__ == '__main__':
    """
    The code below is for testing and visualizing the seam carving functions
    """

    im = cv2.cvtColor(cv2.imread(".\\Figures\\Castle.jpg"), cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    """
    start = time.time()
    im2, seam2 = add_rows(im, e1_colour_numba, 100)
    print(f'Time elapsed: {round(time.time() - start, 2)}')
    imshow(im2)
    plt.show()
    
    start = time.time()
    im2, seam2 = add_rows(im, hog_colour_numba, 100)
    print(f'Time elapsed: {round(time.time() - start, 2)}')
    imshow(im2)
    plt.show()
    """
    e1 = e1_colour_numba(im)

    vertical_seam = find_vertical_seams(e1)[0][0]
    horizontal_seam = find_horizontal_seams(e1)[0][0]

    for row, col in enumerate(vertical_seam):  # This is just to visualize the vertical seam
        im[int(row), int(col)] = (255, 0, 0)

    for col, row in enumerate(horizontal_seam):  # This is just to visualize the horizontal seam
        im[int(row), int(col)] = (255, 0, 0)

    imshow(im)
    plt.show()

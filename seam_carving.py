import time
from math import sqrt

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, types
from numba.extending import overload
from scipy.ndimage import generic_filter
from skimage.io import imshow


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
    sobel3D = np.zeros((3, 3, 3))
    sobel3DT = np.zeros((3, 3, 3))
    sobel3D[0], sobel3D[1], sobel3D[2] = sobel, sobel, sobel
    sobel3DT[0], sobel3DT[1], sobel3DT[2] = sobel.T, sobel.T, sobel.T
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
    return result


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
    sobel3D = np.zeros((3, 3, 3))
    sobel3DT = np.zeros((3, 3, 3))
    sobel3D[0], sobel3D[1], sobel3D[2] = sobel, sobel, sobel
    sobel3DT[0], sobel3DT[1], sobel3DT[2] = sobel.T, sobel.T, sobel.T
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
def segmentation_colour_numba(image):
    pass


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
def find_vertical_seam(e_matrix, amount=1):
    """
    :param e_matrix: The energy representation of an image, should be 2D numpy array
    :param amount: The number of vertical seams to find
    :return:    (0) 2D numpy array, each ROW contains the COLUMN indices of a vertical seam path
                    In case that 'amount' = 1, a 1D array is returned
                (1) the total energy of the minimal energy path
    """
    height, width = e_matrix.shape
    path_e, path = np.zeros((height, width)), np.zeros((height, width))
    # path_e will store the energy of the minimum energy path available for each pixel,
    # path will store the column index of the pixel above it with the minimal energy path
    path_e[0] = e_matrix[0]
    # Find the minimum energy path (top to bottom) for each pixel
    for row_idx in range(1, height):
        for col_idx in range(width):
            min_nbr_idx = max(0, np.argmin(
                e_matrix[row_idx - 1, max(0, col_idx - 1):min(col_idx + 2, width)]) + col_idx - 1)
            path[row_idx, col_idx] = min_nbr_idx
            path_e[row_idx, col_idx] = e_matrix[row_idx, col_idx] + path_e[row_idx - 1, min_nbr_idx]

    # Backtrack the minimal path from bottom to top
    seam_path = np.zeros((amount, height))  # Will store the row, col indices for pixels in the lowest energy path
    col_idces = path_e[-1].argsort()[:amount]  # Find the 'amount' pixels with lowest total energy path in the last row
    seam_path[:, -1] = col_idces
    for idx, col_idx in enumerate(col_idces):  # For each of the selected starting points
        for row_idx in range(height - 1, 0, -1):  # Traverse from bottom of paths to top following minimal entropy path
            seam_path[idx, row_idx - 1] = path[row_idx, col_idx]
            col_idx = int(path[row_idx - 1, col_idx])
    return seam_path, np.min(path_e[-1])


def find_horizontal_seam(e_matrix, amount=1):
    """
    :param e_matrix: The energy of an image, should be a 2D numpy array
    :param amount: The number of vertical seams to find
    :return:    (0) 2D numpy array, each ROW contains the ROW indices of a vertical seam path
                    In case that 'amount' = 1, a 1D array is returned
                (1) the total energy of the minimal energy path
    """
    return find_vertical_seam(e_matrix.T, amount)


def remove_column(image, energy_matrix):
    """
    :param image: RGB image as 3D numpy array
    :param energy_matrix: The energy of an image, should be a 2D numpy array
    :return:    (0) the reduced image
                (1) the values of the pixels that were reduced
                (2) the ROW indices of the pixels that were removed (seam)
    """
    seam = find_vertical_seam(energy_matrix)[0][0]
    values = image[np.arange(image.shape[1]) == np.array(seam).reshape(seam.shape[0], 1)].reshape(image.shape[0], 3)
    image = image[np.arange(image.shape[1]) != np.array(seam).reshape(seam.shape[0], 1)].reshape(image.shape[0], -1, 3)
    return image, values, seam


@njit
def remove_column_numba(image, energy_matrix):
    """
    :param image: RGB image as 3D numpy array
    :param energy_matrix: The energy of an image, should be a 2D numpy array
    :return:    (0) the reduced image
                (1) the values of the pixels that were reduced
                (2) the ROW indices of the pixels that were removed (seam)
    """
    seam = find_vertical_seam(energy_matrix)[0][0]
    values = np.zeros((image.shape[0], 3))
    new_image = np.zeros((image.shape[0], image.shape[1] - 1, 3))
    for row in range(len(seam)):
        seam_row_idx = int(seam[row])
        values[row] = image[row, seam_row_idx]
        new_image[row, :seam_row_idx] = image[row, :seam_row_idx]
        new_image[row, seam_row_idx:] = image[row, seam_row_idx + 1:]
    return new_image.astype(np.uint8), values, seam


@njit
def remove_row_numba(image, energy_matrix):
    """
    :param image: RGB image as 3D numpy array
    :param energy_matrix: The energy of an image, should be a 2D numpy array
    :return:    (0) the reduced image
                (1) the values of the pixels that were reduced
                (2) the COLUMN indices of the pixels that were removed (seam)
    """
    rows, cols, colours = image.shape
    image_T = np.zeros((cols, rows, colours))
    new_image = np.zeros((rows - 1, cols, colours))
    image_T[:, :, 0], image_T[:, :, 1], image_T[:, :, 2] = image[:, :, 0].T, image[:, :, 1].T, image[:, :, 2].T
    new_image_T, values, seam = remove_column_numba(image_T, energy_matrix.T)
    new_image[:, :, 0], new_image[:, :, 1], new_image[:, :, 2] = new_image_T[:, :, 0].T, new_image_T[:, :,
                                                                                         1].T, new_image_T[:, :, 2].T
    return new_image, values, seam


# Passing functions to numba compilated code is quite hard and complicated, so we won't numba compile this function
def remove_rows_and_cols(image, energy_function, rows_to_remove=0, cols_to_remove=0):
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
    while rows_to_remove > 0 and cols_to_remove > 0:
        seam = []
        values = []
        order = []  # True = row, False = column
        energy_matrix = energy_function(image)
        row_seam = find_vertical_seam(energy_matrix)[1]
        column_seam = find_horizontal_seam(energy_matrix)[1]
        if row_seam[1] <= column_seam[1]:
            image, new_values, new_seam = remove_row_numba(image, energy_matrix)
            order.append(True)
            rows_to_remove -= 1
        else:
            image, new_values, new_seam = remove_column_numba(image, energy_matrix)
            order.append(False)
            cols_to_remove -= 1
        seam.append(new_seam)
        values.append(new_values)

    while rows_to_remove > 0:  # The columns have already been removed
        energy_matrix = energy_function(image)
        image, new_values, new_seam = remove_column_numba(image, energy_matrix)
        order.append(True)
        seam.append(new_seam)
        values.append(new_values)
        rows_to_remove -= 1

    while cols_to_remove > 0:  # The rows have already been removed
        energy_matrix = energy_function(image)
        image, new_values, new_seam = remove_column_numba(image, energy_matrix)
        order.append(False)
        seam.append(new_seam)
        values.append(new_values)
        cols_to_remove -= 1

    return image, values, seam, order



def add_column_numba(image, seam):
    """
    :param image: RGB image as 3D numpy array
    :param seam: The seam where the addition should happen, 1D numpy array
    :return:    (0) the increased image
                (1) the ROW indices of the pixels that were added (seam)
    """
    new_image = np.zeros((image.shape[0], image.shape[1] + 1, 3))
    for row in range(len(seam)):
        seam_row_idx = int(seam[row])
        if not seam_row_idx:  # If the index = 0
            seam_row_idx += 1
        new_values = image[row, seam_row_idx-1]/2 + image[row, seam_row_idx]/2  # Divide before addition to prevent integer overflow
        new_image[row, :seam_row_idx] = image[row, :seam_row_idx]
        new_image[row, seam_row_idx] = new_values
        new_image[row, seam_row_idx+1:] = image[row, seam_row_idx:]
    return new_image.astype(np.uint8)


@njit
def add_row_numba(image, seam):
    """
    :param image: RGB image as 3D numpy array
    :param energy_matrix: The energy of an image, should be a 2D numpy array
    :return:    (0) the increased image
                (1) the COLUMN indices of the pixels that were added (seam)
    """

    rows, cols, colours = image.shape
    image_T = np.zeros((cols, rows, colours))
    new_image = np.zeros((rows + 1, cols, colours))
    image_T[:, :, 0], image_T[:, :, 1], image_T[:, :, 2] = image[:, :, 0].T, image[:, :, 1].T, image[:, :, 2].T
    new_image_T = add_column_numba(image_T, seam)
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
    while columns_to_add > 0:
        if columns_to_add >= image.shape[0]//2:
            amount = image.shape[0]//2
        else:
            amount = columns_to_add
        energy_matrix = energy_function(image)
        new_seams = find_vertical_seam(energy_matrix, amount=amount)[0]
        for seam in new_seams:
            image = add_column_numba(image, seam)
        if seams:
            seams += new_seams
        else:
            seams = new_seams
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


if __name__ == '__main__':
    """
    The code below is for testing and visualizing the seam carving functions
    """

    im = cv2.cvtColor(cv2.imread(".\\Figures\\Castle.jpg"), cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

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

    e1 = e1_colour_numba(im)

    vertical_seam = find_vertical_seam(e1)[0][0]
    horizontal_seam = find_horizontal_seam(e1)[0][0]

    for row, col in enumerate(vertical_seam):  # This is just to visualize the vertical seam
        im[int(row), int(col)] = (255, 0, 0)

    for col, row in enumerate(horizontal_seam):  # This is just to visualize the horizontal seam
        im[int(row), int(col)] = (255, 0, 0)

    imshow(im)
    plt.show()

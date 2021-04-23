import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import generic_filter
from scipy.signal import convolve, convolve2d
from skimage.io import imshow
from skimage.feature import hog as histogram_of_gradients
from skimage.filters.rank import entropy as skimage_entropy
import timeit

def e1_gray(image):
    """
    :param image: a grayscale image as 2D numpy array
    :return: a 2D numpy array with the same size as 'image', this array holds e1 energy for every pixel in the image.
    This error represents the sum of derivative in x and y direction for each pixel, this has been calculated via the
    sobel kernel.
    """
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return np.abs(convolve2d(image, sobel, 'same')) + np.abs(convolve(image, sobel.T, 'same'))


def e1_colour(image):
    """
    :param image: a color image as 3D numpy array
    :return: a 2D numpy array holding the e1 energy for every pixel in the image.
    """
    return e1_gray(image[:, :, 0]) / 3 + e1_gray(image[:, :, 1]) / 3 + e1_gray(image[:, :, 2]) / 3


def entropy_helper(grid):
    """
    Helper function for the entropy calculation
    :param grid: an nxm grid
    :return: the Shannon entropy of the grid
    """
    probabilities = np.bincount(np.int64(grid))/grid.size
    probabilities = probabilities[probabilities != 0]
    return -(probabilities * np.log2(probabilities)).sum()


def entropy(image):
    """
    :param image: 2D image
    :return: entropy of the 2D image for each pixel.
    The entropy is calculated with a 9x9 grid, as indicated in Avidan & Shamir (2007).
    """
    return generic_filter(np.array(image), entropy_helper, size=(9, 9), output=np.zeros(image.shape))


# This is a slower implementation
"""
def entropy(image):
    rows, cols = np.shape(image)
    result = np.zeros((rows, cols))
    for row_index in range(rows):
        row_start, row_end = max(0, row_index - 4), min(row_index + 5, rows)
        for col_index in range(cols):
            col_start, col_end = max(0, col_index - 4), min(col_index + 5, cols)
            grid = image[row_start:row_end, col_start:col_end].flatten()
            probabilities = np.bincount(grid) / grid.size
            probabilities = probabilities[probabilities != 0]
            result[row_index, col_index] = -(probabilities * np.log2(probabilities)).sum()
    return result
"""


def entropy_gray(image):
    return e1_gray(image) + entropy(image)


def entropy_colour(image):
    # I should not divide the e1_colour by 255 but without it it has values >1000, while entropy goes to ~6...
    return e1_colour(image)/255 + entropy(image[:, :, 0]) / 3 + entropy(image[:, :, 1]) / 3 + entropy(image[:, :, 2]) / 3


def segmentation(image):
    pass


# Too slow to practically use, take ~1 minute to complete once for a 250x200 image (at least on my laptop).
# I tried to solve this via a combination of matrix manipulations and filters instead of for loops
# but cant find a solution. Also unsure whether my solution is correct,
# it's hard to find a good function to compare with.
def hog(image):
    """
    :param image:
    :return:
    """
    rows, cols = image.shape
    sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    dx = np.abs(convolve2d(image, sobel, 'same'))
    dy = np.abs(convolve(image, sobel.T, 'same'))
    gradient = np.sqrt(np.square(dx) + np.square(dy))
    angle = (np.arctan2(dy, dx) * 360 / np.pi) % 180
    bin_num, bin_spread = (angle // 20).astype(np.uint8), angle % 20 * 0.05
    result = np.zeros(image.shape)
    for row_idx, gradient_row in enumerate(gradient):
        row_end = min(row_idx + 11, rows)
        for col_idx, gradient_cell in enumerate(gradient_row):
            bins = np.zeros(9)
            col_end = min(col_idx + 11, cols)
            gradient_grid = gradient[row_idx:row_end, col_idx:col_end].flatten()
            bin_num_grid = bin_num[row_idx:row_end, col_idx:col_end].flatten()
            bin_spread_grid = bin_spread[row_idx:row_end, col_idx:col_end].flatten()
            for idx, bin in enumerate(bin_num_grid):
                bins[bin] += gradient_grid[idx] * (1 - bin_spread_grid[idx])
                bins[(bin + 1) % 9] += gradient_grid[idx] * bin_spread_grid[idx]
            result[row_idx, col_idx] += max(bins)
    return result


def hog_gray(image):
    return e1_gray(image) / hog(image)


def hog_col(image):
    return e1_colour(image) / np.sum((hog(image[:, :, 0]), hog(image[:, :, 1]), hog(image[:, :, 2])))


def find_vertical_seam(e_matrix):
    """
    :param e_matrix: The error representation of a grayscale image, should be 2D numpy array
    :return: 1D numpy array containing the col indices of the vertical seam path
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
    seam_path = np.zeros(height)  # Will store the row, col indices for pixels in the lowest energy path
    col_idx = np.argmin(path_e[-1])  # Find the pixel with lowest total energy path in the last row, here we start
    seam_path[-1] = col_idx
    for row_idx in range(height - 1, 0, -1):  # Traverse from bottom of paths to top following minimal entropy path
        seam_path[row_idx - 1] = path[row_idx, col_idx]
        col_idx = int(path[row_idx - 1, col_idx])
    return seam_path  # Returns an array containing the ordered col indices of the vertical seam path


def find_horizontal_seam(e_matrix):
    """
    :param entropy_image: The entropy representation of a grayscale image, should be a 2D numpy array
    :return: 1D array containing the row indices of the horizontal seam path
    """
    return find_vertical_seam(e_matrix.T)


def remove_column(image, error_function, number_of_columns=1):
    """
    :param image: RGB image as 3D numpy array
    :param error_function: one of the error functions defined above
    :return: three items: the reduced image, the values of the pixels that were reduced,
    the row indices of the pixels that were removed
    """
    seam = np.zeros((number_of_columns, image.shape[0]))
    values = np.zeros((number_of_columns, image.shape[0], 3))
    for x in range(number_of_columns):
        seam[x] = find_vertical_seam(error_function(image))
        values[x] = image[np.arange(image.shape[1]) == np.array(seam[x])[:, None]].reshape(image.shape[0], 3)
        image = image[np.arange(image.shape[1]) != np.array(seam[x])[:, None]].reshape(image.shape[0], -1, 3)
    return image, values, seam


# Below is a recursive version of the remove_column function. Speed test has shown that it is ~10% slower
# than the iterative version. Probably due to the use of np.concatenate
'''
def remove_column(image, error_function, number_of_columns=1):
    """
    :param image: RGB image as 3D numpy array
    :param error_function: one of the error functions defined above
    :return: three items: the reduced image, the values of the pixels that were reduced,
    the row indices of the pixels that were removed
    """
    if number_of_columns == 1:
        seam = find_vertical_seam(error_function(image))
        values = image[np.arange(image.shape[1]) == np.array(seam)[:, None]].reshape(image.shape[0], 3)
        image = image[np.arange(image.shape[1]) != np.array(seam)[:, None]].reshape(image.shape[0], -1, 3)
        return image, values, seam
    else:
        image, previous_values, previous_seam = remove_column(image, error_function, number_of_columns-1)
        seam = find_vertical_seam(error_function(image))
        values = image[np.arange(image.shape[1]) == np.array(seam)[:, None]].reshape(image.shape[0], 3)
        image = image[np.arange(image.shape[1]) != np.array(seam)[:, None]].reshape(image.shape[0], -1, 3)
    return image, np.concatenate((values, previous_values)), np.concatenate((seam, previous_seam))
'''


def remove_row(image, error_function, number_of_rows=1):
    """
    :param image: RGB image as 3D numpy array
    :param error_function: one of the error functions defined above
    :return: three items: the reduced image, the values of the pixels that were reduced,
    the col indices of the pixels that were removed
    """
    transposed_image, values, seam = remove_column(np.transpose(image, (1, 0, 2)), error_function, number_of_rows)
    return np.transpose(transposed_image, (1, 0, 2)), values, seam


if __name__ == '__main__':
    """
    The code below is for testing and visualizing the seam carving functions
    """

    im = cv2.cvtColor(cv2.imread(".\\Figures\\River.png"), cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


    e1 = e1_colour(im)
    imshow(e1, cmap='gray')
    plt.show()

    entropy_im = entropy_colour(im)
    imshow(entropy_im, cmap='gray')
    plt.show()

    hog_im = hog_col(im)
    imshow(hog_im, cmap='gray')
    plt.show()

    vertical_seam = find_vertical_seam(hog_im)
    horizontal_seam = find_horizontal_seam(hog_im)
    
    for row, col in enumerate(vertical_seam):  # This is just to visualize the vertical seam
        im[int(row), int(col)] = (255, 0, 0)

    for col, row in enumerate(horizontal_seam):  # This is just to visualize the horizontal seam
        im[int(row), int(col)] = (255, 0, 0)

    imshow(im)
    plt.show()


import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve, convolve2d
from skimage.io import imshow


def e1_gray(image):
    """
    :param image: a grayscale image as 2D numpy array
    :return: a 2D numpy array with the same size as 'image', this array holds e1 energy for every pixel in the image.
    This error represents the sum of derivative in x and y direction for each pixel, this has been calculated via the
    sobol kernel.
    """
    sobol = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return np.abs(convolve2d(image, sobol, 'same')) + np.abs(convolve(image, sobol.T, 'same'))


def e1_col(image):
    """
    :param image: a color image as 3D numpy array
    :return: a 2D numpy array holding the e1 energy for every pixel in the image.
    """
    return e1_gray(image[:, :, 0]) + e1_gray(image[:, :, 1]) + e1_gray(image[:, :, 2])


def entropy(image):
    """
    :param image:
    :return:
    The entropy is calculated with a 9x9 grid, as indicated in Avidan & Shamir (2007).
    """
    # Not yet operational
    rows, cols = np.shape(image)
    result = e1_col(image)
    for row_index in range(rows):
        row_start, row_end = max(0, row_index - 4), min(row_index + 5, rows)
        for col_index in range(cols):
            col_start, col_end = max(0, col_index - 4), min(col_index + 5, cols)
            grid = image[row_start:row_end, col_start:col_end]
            result[row_index, col_index] -= 0  # Replace 0 by entropy of grid
    return result


def segmentation(image):
    pass


def HoG(image):
    pass


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


def remove_column(image, error_function):
    """
    :param image: RGB image as 3D numpy array
    :param error_function: one of the error functions defined above
    :return: three items: the reduced image, the values of the pixels that were reduced,
    the row indices of the pixels that were removed
    """
    seam = find_vertical_seam(error_function(image))
    values = image[np.arange(image.shape[1]) == np.array(seam)[:, None]].reshape(image.shape[0], 3)
    image = image[np.arange(image.shape[1]) != np.array(seam)[:, None]].reshape(image.shape[0], -1, 3)
    return image, values, seam

def remove_row(image, error_function):
    """
    :param image: RGB image as 3D numpy array
    :param error_function: one of the error functions defined above
    :return: three items: the reduced image, the values of the pixels that were reduced,
    the col indices of the pixels that were removed
    """
    # Probably faster if I can get this to work with 'find_horizontal_seam' but this works in the mean time
    transposed_image, values, seam = remove_column(np.transpose(image, (1, 0, 2)), error_function)
    return np.transpose(transposed_image, (1, 0, 2)), values, seam


if __name__ == '__main__':
    """
    The code below is for testing and visualizing the seam carving functions
    """

    im = cv2.cvtColor(cv2.imread(".\\Figures\\Castle.jpg"), cv2.COLOR_BGR2RGB)

    print(im.shape)
    for x in range(50):
        im = remove_column(im, e1_col)[0]
        print(f'Removed {x+1} columns')
    print(im.shape)
    imshow(im)
    plt.show()

    im = cv2.cvtColor(cv2.imread(".\\Figures\\Castle.jpg"), cv2.COLOR_BGR2RGB)
    e1_c = e1_col(im)

    vertical_seam = find_vertical_seam(e1_c)
    horizontal_seam = find_horizontal_seam(e1_c)
    
    for row, col in enumerate(vertical_seam):  # This is just to visualize the vertical seam
        im[int(row), int(col)] = (255, 0, 0)

    for col, row in enumerate(horizontal_seam):  # This is just to visualize the horizontal seam
        im[int(row), int(col)] = (255, 0, 0)

    imshow(im)
    plt.show()

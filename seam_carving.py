import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.io import imread, imshow
import skimage.color as ski_col

def e1(image):
    """
    :param image: RGB image
    :return: Entropy of the (grayscale) image
    Takes an image as input and returns its entropy based on the 9x9 grid around each pixel.
    Will eventually have to write the 'entropy' function ourselves probably since this is 'too simple' for us to just
    'copy-paste'.
    """
    return entropy(ski_col.rgb2gray(image), disk(9))  # I assume disk(9) will be a 9x9 grid but it's poorly documented


def HoG(image):
    pass


def find_vertical_seam(entropy_image):
    """
    :param entropy_image: The entropy representation of an image (2-D), should be numpy array
    :return: 2-D array containing the row, column indices of the vertical seam path
    """
    height, width = entropy_image.shape
    paths = np.zeros((height, width))
    paths[0] = entropy_image[0]
    for row_idx, row in enumerate(entropy_image):
        paths[row_idx, 0] = row[0] + min(paths[row_idx-1, 0], paths[row_idx-1, 1])  # First value in row
        for col_idx in range(1, width-1):
            paths[row_idx, col_idx] = row[col_idx] + min(paths[row_idx-1, col_idx-1], paths[row_idx-1, col_idx], paths[row_idx-1, col_idx+1])
        paths[row_idx, -1] = row[-1] + min(paths[row_idx - 1, -2], paths[row_idx - 1, -1])  # Last value in row

    seam_path = np.zeros((height, 2))
    position = np.where(paths[-1] == np.amin(paths[-1]))[0][0]
    seam_path[-1] = height-1, position
    for row_idx in range(-1, -height, -1):  # Traverse from bottom of paths to top following minimal entropy path
        if not position:
            neighbors, subtraction = paths[row_idx - 1, 0:2], 0
        elif position == width-1:
            neighbors, subtraction = paths[row_idx - 1, -2::], 1
        else:
            neighbors, subtraction = paths[row_idx-1, position-1:position+2], 1
        minimum_neighbor = np.where(neighbors == np.amin(neighbors))[0][0] + position - subtraction
        seam_path[row_idx-1] = row_idx+len(paths)-1, minimum_neighbor
        position = minimum_neighbor
    return seam_path  # Returns a 2-D array containing the row, column indices of the vertical seam path

def find_horizontal_seam(entropy_image):
    """
    :param entropy_image: The entropy representation of an image (2-D), should be numpy array
    :return: 2-D array containing the row, column indices of the horizontal seam path
    """
    return find_vertical_seam(entropy_image.T)[:, [1,0]]



"""
The code below is for testing and visualizing the seam carving functions
"""

im = imread(".\\Figures\\Castle.jpg")
entropy_im = e1(im)
vertical_seam = find_vertical_seam(entropy_im)
horizontal_seam = find_horizontal_seam(entropy_im)

for row_col in vertical_seam:  # This is just to visualize the vertical seam
    im[int(row_col[0]), int(row_col[1])] = (0,0,0)

for row_col in horizontal_seam:  # This is just to visualize the horizontal seam
    im[int(row_col[0]), int(row_col[1])] = (0,0,0)

imshow(im)
plt.show()

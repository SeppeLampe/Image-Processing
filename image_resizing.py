import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve, convolve2d
from skimage.io import imshow
import argparse
import seam_carving as sc
from SeamImage import SeamImage
import os

def resize(image, width_diff, height_diff, energy_function):
    """
    :param image: RGB image as 3D numpy array
    :param width_diff: the number of columns to be added (if negative) or removed (if positive)
    :param height_diff: the number of rows to be added (if negative) or removed (if positive)
    :return: the modified image
    """

    # Create a SeamImage object
    my_seam_image = SeamImage(image)

    # Energy function can be e1_colour_numba, entropy_colour_numba or hog_colour_numba
    if height_diff >= 0 and width_diff >= 0:
        my_seam_image.remove_rows_and_cols(energy_function=energy_function, rows=height_diff, cols=width_diff) # Remove both columns and rows
    elif height_diff < 0 and width_diff < 0:
        my_seam_image.add_rows_and_cols(energy_function=energy_function, rows=-height_diff, cols=-width_diff) # Add both columns and rows
    elif height_diff >= 0 and width_diff < 0:
        my_seam_image.remove_rows_and_cols(energy_function=energy_function, rows=height_diff, cols=0) # Remove rows
        my_seam_image.add_rows_and_cols(energy_function=energy_function, rows=0, cols=-width_diff) # Add columns
    else:
        my_seam_image.remove_rows_and_cols(energy_function=energy_function, rows=0, cols=width_diff) # Remove columns
        my_seam_image.add_rows_and_cols(energy_function=energy_function, rows=-height_diff, cols=0) # Add rows

    print(my_seam_image.image.shape)

    return my_seam_image.image



if __name__ == '__main__':
    """
    This code is for resizing images using seam carving
    """
    parser = argparse.ArgumentParser(description='This code is for resizing images using seam carving')

    parser.add_argument("--input_image", default='Figures\Dolphin.jpg', type=str, help='Specify the input image to be resized')
    parser.add_argument("--output_image_width", default=255, type=int, help='Specify the width of the output image in pixels')
    parser.add_argument("--output_image_height", default=220, type=int, help='Specify the height of the output image in pixels')
    parser.add_argument("--energy_function", default='e1_colour_numba', type=str, help='Specify the energy function to be used for seam carving. Can be e1_colour_numba, entropy_colour, entropy_colour_numba, hog_colour_numba')

    args = parser.parse_args()

    im = cv2.cvtColor(cv2.imread(args.input_image), cv2.COLOR_BGR2RGB)

    input_width = im.shape[1]
    input_height = im.shape[0]

    width_diff = input_width - args.output_image_width
    height_diff = input_height - args.output_image_height

    if args.energy_function == 'e1_colour_numba':
        energy_function = sc.e1_colour_numba
    elif args.energy_function == 'entropy_colour':
        energy_function = sc.entropy_colour
    elif args.energy_function == 'entropy_colour_numba':
        energy_function = sc.entropy_colour_numba
    elif args.energy_function == 'hog_colour_numba':
        energy_function = sc.hog_colour_numba

    output_with_sc = resize(im, width_diff, height_diff, energy_function)
    output_regular = cv2.resize(im, (args.output_image_width, args.output_image_height))
    plt.imshow(output_with_sc)
    plt.show()
    plt.close()

    _, ext = os.path.splitext(args.input_image)
    output_file_name_with_sc = args.input_image.replace(ext, '_resized_with_seamcarving.png')
    output_file_name_regular = args.input_image.replace(ext, '_resized_without_seamcarving.png')

    output_image_bgr_sc = cv2.cvtColor(output_with_sc, cv2.COLOR_RGB2BGR) #converted again to use in imwrite
    cv2.imwrite(output_file_name_with_sc, output_image_bgr_sc)

    output_image_bgr_regular = cv2.cvtColor(output_regular, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file_name_regular, output_image_bgr_regular)


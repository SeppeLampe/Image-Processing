import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve, convolve2d
from skimage.io import imshow
import argparse
from seam_carving import e1_col, remove_column, remove_row, find_vertical_seam, find_horizontal_seam
import os

def resize(image, width_diff, height_diff):
    """
    :param image: RGB image as 3D numpy array
    :param width_diff: the number of columns to be added or removed
    :param height_diff: the number of rows to be added or removed
    :return: the reduced image
    """
    
    # First remove columns or add columns depending on the case
    for x in range(abs(width_diff)):
        if width_diff > 0:
            image = remove_column(image, e1_col)[0]
            print(f'Removed {x+1} columns')
        else:
            #image = add_column(image, e1_col)[0]
            print(f'Added {x+1} columns')

    print(image.shape)

    # Remove rowss or add rows depending on the case
    for x in range(abs(height_diff)):
        if height_diff > 0:
            image = remove_row(image, e1_col)[0]
            print(f'Removed {x+1} rows')
        else:
            #image = add_row(image, e1_col)[0]
            print(f'Added {x+1} rows')

    print(image.shape)

    return image



if __name__ == '__main__':
    """
    This code is for resizing images using seam carving
    """
    parser = argparse.ArgumentParser(description='This code is for resizing images using seam carving')

    parser.add_argument("--input_image", default='Broadway_tower_edit.jpg', type=str, help='Specify the input image to be resized')
    parser.add_argument("--output_image_width", default=1400, type=int, help='Specify the width of the output image in pixels')
    parser.add_argument("--output_image_height", default=968, type=int, help='Specify the height of the output image in pixels')

    args = parser.parse_args()

    im = cv2.cvtColor(cv2.imread(args.input_image), cv2.COLOR_BGR2RGB)

    input_width = im.shape[1]
    input_height = im.shape[0]

    width_diff = input_width - args.output_image_width
    height_diff = input_height - args.output_image_height

    output_with_sc = resize(im, width_diff, height_diff)
    output_regular = cv2.resize(im, (args.output_image_width, args.output_image_height))
    
#    plt.imshow(output_image)
#    plt.show()
#    plt.close()

    _, ext = os.path.splitext(args.input_image)
    output_file_name_with_sc = args.input_image.replace(ext, '_resized_with_seamcarving.png')
    output_file_name_regular = args.input_image.replace(ext, '_resized_without_seamcarving.png')

    output_image_bgr_sc = cv2.cvtColor(output_with_sc, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file_name_with_sc, output_image_bgr_sc)

    output_image_bgr_regular = cv2.cvtColor(output_regular, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file_name_regular, output_image_bgr_regular)


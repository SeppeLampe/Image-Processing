import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow
import SeamImage
import seam_carving
import argparse
from image_resizing import resize
#from seam_carving import e1_col, remove_column, remove_row, find_vertical_seam, find_horizontal_seam
#from image_resizing import resize
import os

def object_removal(image, mask):

    '''    remove_width = np.abs(left-right)
        remove_height = np.abs(top-bottom)'''


    masked_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)



    shape = np.where( mask == 1)
    max_width = max( shape[0] ) - min( shape[0] )+1
    max_height = max( shape[1] ) - min( shape[1] )+1
    print(max_width)
    print(max_height)
    '''
    if( max_width>max_height ):
        removed_image = resize(masked_image, max_width,0)
    else:
        removed_image = resize(masked_image, 0, max_height)'''

    return  masked_image


if __name__ == "__main__":
    """
       The code below is for testing and visualizing the object removal function
       """

    im = cv2.cvtColor(cv2.imread(".\\Figures\\Castle.jpg"), cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    H, W = im_gray.shape
    rect_width = 50
    rect_height = 80
    top, left = 50,300
    mask = np.zeros( (H, W) )
    for i in range(rect_height):
        for j in range(rect_width):
            mask[top + i, left+j ] = 1

    mask = cv2.normalize(mask.astype('uint8'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    plt.figure(1)
    plt.subplot(1,3,1), plt.imshow(im), plt.title("Original")
    plt.subplot(1,3,2), plt.imshow(mask, cmap="gray" ), plt.title("Mask")
    plt.subplot(1,3,3), plt.imshow(im_gray, cmap="gray" ), plt.title("Gray")
    plt.show()

'''    masked_im = object_removal( im, mask)
    plt.figure()
    plt.subplot(1,3,1), plt.imshow(im), plt.title("Original")
    plt.subplot(1,3,2), plt.imshow(masked_im ), plt.title("Mask")
    plt.show()'''


'''
    '''



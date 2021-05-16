import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow
import SeamImage
from SeamImage import SeamImage
import seam_carving as sc
from PIL import Image
import image_resizing
from image_resizing import resize
#from seam_carving import e1_col, remove_column, remove_row, find_vertical_seam, find_horizontal_seam
#from image_resizing import resize
import os

def object_removal(image, mask, energy_function=sc.e1_colour_numba):
    '''
    Removes a masked object from the image

    :param image: The original iamge
    :param mask: The mask for the object removal
    :return: The image without the object, and same size as the original
    '''

    #initialiyation for the image
    my_seam_image = SeamImage(image)

    # Calculate the MAX width and height
    shape = np.where( mask == 1)
    max_height= max( shape[0] ) - min( shape[0] )+1
    max_width = max( shape[1] ) - min( shape[1] )+1
    print('Max width: ', max_width)
    print('Max Height: ', max_height)
    print('Image: ', image.shape )
    print('Mask: ', mask.shape )

    # Removal along the narrower dimension
    if( max_width>max_height ):
        my_seam_image.remove_rows_and_cols( energy_function=energy_function, rows=max_height, cols=0, mask=mask)  # Remove columns
        print('Rows adding started')
        #my_seam_image.always_add( energy_function=energy_function, rows=max_height , cols=0)
    else:
        my_seam_image.remove_rows_and_cols(energy_function=energy_function, rows=0, cols=max_width, mask=mask)  # Remove columns
        print('Cols adding started')
        #.add_always_new_cols(energy_function=energy_function, cols=max_width)

    return  my_seam_image.image


if __name__ == "__main__":
    """
       The code below is for testing and visualizing the object removal function
       """

    # Read the image
    im = cv2.cvtColor(cv2.imread(".\\Figures\\Dolphin.jpg"), cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # Open the pre created mask
    mask = cv2.cvtColor(cv2.imread(".\\Figures\\Dolphin_masked_dolphin.jpg"), cv2.COLOR_BGR2RGB )
    mask_gray = cv2.cvtColor(cv2.imread(".\\Figures\\Dolphin_masked_dolphin.jpg"), cv2.COLOR_BGR2GRAY )

    '''
    # Read the image
    im = cv2.cvtColor(cv2.imread(".\\Figures\\Castle.jpg"), cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # Open the pre created mask
    mask = cv2.cvtColor(cv2.imread(".\\Figures\\Castle_masked_person.jpg"), cv2.COLOR_BGR2RGB)
    mask_gray = cv2.cvtColor(cv2.imread(".\\Figures\\Castle_masked_person.jpg"), cv2.COLOR_BGR2GRAY)'''

    # With thresholding generate a reliable binary image
    ret, mask = cv2.threshold(mask, 50, 1, cv2.THRESH_BINARY)
    ret, mask_gray = cv2.threshold(mask_gray, 50, 1, cv2.THRESH_BINARY)

    #calculate the inpaintings
    masked_image = cv2.inpaint(im, mask_gray, 3, cv2.INPAINT_TELEA)
    masked_im_remove = object_removal( im, mask, energy_function=sc.e1_colour_numba)
    print( masked_im_remove.shape)

    # Plot the Basic images
    plt.figure(1)
    plt.subplot(2,3,1), plt.imshow(im), plt.title("Original")
    plt.subplot(2,3,3), plt.imshow(mask[:,:,0] , cmap="gray"), plt.title("Mask")
    plt.subplot(2,3,2), plt.imshow(im_gray, cmap="gray" ), plt.title("Gray")

    # Plot the removal images
    plt.subplot(2,3,4), plt.imshow( sc.e1_colour_numba(im) ), plt.title("Original")
    plt.subplot(2,3,5), plt.imshow(masked_image ), plt.title("Mask")
    plt.subplot(2,3,6), plt.imshow(masked_im_remove ), plt.title("Mask Remove")
    plt.show()



    '''
    # Generate a mask manually
    # Left top of the tower
    H, W = im_gray.shape
    print(im_gray.shape)
    rect_width = 70
    rect_height = 25
    top, left = 0, 365

    # person
    H, W = im_gray.shape
    print(im_gray.shape)
    rect_width = 20
    rect_height = 40
    top, left = 205, 45

    mask = np.zeros( (H, W) )
    for i in range(rect_height):
        for j in range(rect_width):
            mask[top + i, left+j ] = 1
    # Convert the array to image
    mask = cv2.normalize(mask.astype('uint8'), None, 0.0, 1.0, cv2.NORM_MINMAX) # im2double'''


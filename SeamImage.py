import seam_carving as sc
import cv2
from skimage.io import imshow
import matplotlib.pyplot as plt
import time

class SeamImage:
    def __init__(self, location, color=True):
        self.location = location
        # self.image is the current state of the image, will change continuously
        if color:
            self.image = cv2.cvtColor(cv2.imread(location), cv2.COLOR_BGR2RGB)
        else:
            self.image = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
        self.original_shape = self.image.shape

        self.added_seams = []  # Stores np.arrays containing the row/col indices that were added
        self.added_order = []  # Will store True/False based on whether a row or col was added respectively

        self.removed_seams = []  # Stores np.arrays containing the row/col indices that were removed
        self.removed_values = []  # Stores np.arrays containing the values that were removed
        self.removed_order = []  # Will store True/False based on whether a row or col was removed respectively

    def get_image(self):
        return self.image

    def reset(self):
        self.image = cv2.cvtColor(cv2.imread(self.location), cv2.COLOR_BGR2RGB)

    def remove_rows_and_cols(self, energy_function=sc.e1_colour_numba, rows=0, cols=0):
        """
        Removes rows and/or columns to the image
        :param energy_function: an energy function to apply for the seam carving
        :param rows: amount of rows to remove
        :param cols: amount of columns to remove
        :return: Nothing
        """
        while self.added_order and (rows > 0 or cols > 0):
            if self.added_order.pop():  # Removing an earlier added row
                self.image = sc.remove_row_seam_numba(self.image, self.added_seams.pop())[0]
                rows -= 1
            else:  # Removing an earlier added column
                self.image = sc.remove_column_seam_numba(self.image, self.added_seams.pop())[0]
                cols -= 1

        # There is a possibility we removed some rows too much because they (rows/columns) were in ordered order
        if rows < 0:
            self.image, new_seams = sc.add_rows(self.image, energy_function, -rows)
            self.added_seams.extend(new_seams)

        # There is a possibility we removed some columns too much because they (rows/columns) were in ordered order
        if cols < 0:
            self.image, new_seams = sc.add_columns(self.image, energy_function, -cols)
            self.added_seams.extend(new_seams)

        # Remove the amount of rows and columns from the image that remain to be removed
        if rows > 0 or cols > 0:
            self.image, new_values, new_seams, new_orders = sc.remove_rows_and_cols(self.image, energy_function, rows, cols)
            self.removed_values.extend(new_values)
            self.removed_seams.extend(new_seams)
            self.removed_order.extend(new_orders)

    def add_rows_and_cols(self, energy_function=sc.e1_colour, rows=0, cols=0):
        """
        Adds rows and/or columns to the image
        :param energy_function: an energy function to apply for the seam carving
        :param rows: amount of rows to add
        :param cols: amount of columns to add
        :return: Nothing
        """
        while self.removed_order and (rows > 0 or cols > 0):
            if self.removed_order.pop():  # Adding an earlier removed row
                self.image = sc.reconstruct_row_seam_numba(self.image, self.removed_values.pop(), self.removed_seams.pop())
                rows -= 1
            else:  # Adding an earlier removed column
                self.image = sc.reconstruct_column_seam_numba(self.image, self.removed_values.pop(), self.removed_seams.pop())
                cols -= 1

        # There is a possibility we added some rows or columns too much because they (rows/columns) were in ordered order in removed_seams
        if rows < 0 or cols < 0:
            self.image, new_values, new_seams, new_orders = sc.remove_rows_and_cols(self.image, energy_function, -rows,
                                                                                    -cols)
            self.removed_values.extend(new_values)
            self.removed_seams.extend(new_seams)
            self.removed_order.extend(new_orders)

        # Add the amount of rows to the image that remain to be added
        if rows > 0:
            self.image, new_seams = sc.add_rows(self.image, energy_function, rows)
            self.added_seams.extend(new_seams)
            self.added_order.extend([True for _ in range(rows)])

        # Add the amount of columns to the image that remain to be added
        if cols > 0:
            self.image, new_seams = sc.add_columns(self.image, energy_function, cols)
            self.added_seams.extend(new_seams)
            self.added_order.extend([False for _ in range(cols)])

    def resize(self, energy_function=sc.e1_colour_numba, height=0, width=0):
        """
        Resizes the image to the specified format
        :param energy_function: an energy function to apply for the seam carving
        :param height: The height of the new image
        :param width: The width of the new image
        :return: Nothing
        """
        rows, cols = height-self.image.shape[0], width-self.image.shape[1]
        # Add rows and columns
        if rows >= 0 and cols >= 0:
            self.add_rows_and_cols(energy_function=energy_function, rows=rows, cols=cols)
        # Remove rows and columns
        elif rows <= 0 and cols <= 0:
            self.remove_rows_and_cols(energy_function=energy_function, rows=-rows, cols=-cols)
        # Add rows, remove columns
        elif rows >= 0 >= cols:
            self.remove_rows_and_cols(energy_function=energy_function, rows=rows, cols=0)
            self.add_rows_and_cols(energy_function=energy_function, rows=0, cols=-cols)
        # Remove rows, add columns
        else:
            self.remove_rows_and_cols(energy_function=energy_function, rows=0, cols=cols)
            self.add_rows_and_cols(energy_function=energy_function, rows=-rows, cols=0)

    def remove_mask(self, mask, energy_function=sc.e1_colour_numba, keep_shape=False):
        """
        Removes a mask from an image
        :param mask: a binary 2D array where the area to be removed is 1, the rest 0
        :param energy_function: an energy function to apply for the seam carving
        :param keep_shape: Boolean to indicate whether the original shape should be kept
        :return: Nothing
        """
        original_rows, original_cols = self.image.shape[0], self.image.shape[1]
        self.image = sc.remove_mask(self.image, energy_function=energy_function, mask=mask)
        if keep_shape:
            self.resize(energy_function, original_rows, original_cols)

    def amplify_content(self, amp_factor=1.2):
        """
        First scales the image in size (content-unaware) given by amp_factor
        Then content-aware resize it back to its original shape
        :param amp_factor: amplification factor for the image
        :return: Nothing
        """
        # Simply scale the image first
        self.image = cv2.resize(self.image, (int(self.image.shape[1]*amp_factor), int(self.image.shape[0]*amp_factor)))
        
        # Rescale it back to original size with seam carving
        self.remove_rows_and_cols(rows=int(self.image.shape[0] - self.original_shape[0]), cols=int(self.image.shape[1] - self.original_shape[1]))


if __name__ == '__main__':
    my_seam_image = SeamImage(location=".\\Figures\\Castle.jpg")
    imshow(my_seam_image.image)
    plt.show()
    my_seam_image.amplify_content(1.2)
    output_amplified = cv2.cvtColor(my_seam_image.image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('.\\Figures\\Castle_amplified.jpg', output_amplified)
    
    mask = cv2.imread('.\\Figures\\Castle_masked_person.jpg', cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 50, 1, cv2.THRESH_BINARY)
    start = time.time()
    my_seam_image.remove_mask(mask)
    print(f'Time elapsed: {round(time.time() - start, 2)}')
    imshow(my_seam_image.image)
    plt.show()

    start = time.time()
    my_seam_image.remove_rows_and_cols(rows=75, cols=150)
    print(f'Time elapsed: {round(time.time() - start, 2)}')
    imshow(my_seam_image.image)
    plt.show()

    start = time.time()
    my_seam_image.remove_rows_and_cols(rows=75, cols=150)
    print(f'Time elapsed: {round(time.time() - start, 2)}')
    imshow(my_seam_image.image)
    plt.show()

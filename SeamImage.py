import seam_carving as sc
import cv2
from skimage.io import imshow
import matplotlib.pyplot as plt
import time
import numpy as np

class SeamImage:
    def __init__(self, image):
        self.image = image  # The current state of the image, will change continuously
        self.original_shape = image.shape

        self.added_seams = []  # Stores np.arrays containing the row/col indices that were added
        self.added_order = []  # Will store True/False based on whether a row or col was added respectively

        self.removed_seams = []  # Stores np.arrays containing the row/col indices that were removed
        self.removed_values = []  # Stores np.arrays containing the values that were removed
        self.removed_order = []  # Will store True/False based on whether a row or col was removed respectively

    def remove_rows_and_cols(self, energy_function=sc.e1_colour_numba, rows=0, cols=0, mask=False):
        while self.added_order and (rows > 0 or cols > 0):
            if self.added_order.pop():  # Removing an earlier added row
                self.image = sc.remove_row_seam_numba(self.image, self.added_seams.pop(), mask )[0]
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
            self.image, new_values, new_seams, new_orders = sc.remove_rows_and_cols(self.image, energy_function, rows, cols, mask=mask)
            self.removed_values.extend(new_values)
            self.removed_seams.extend(new_seams)
            self.removed_order.extend(new_orders)

    def add_rows_and_cols(self, energy_function=sc.e1_colour, rows=0, cols=0):
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

    # Always add new rows to the image, needed for the removal
    def add_always_new_rows(self, energy_function=sc.e1_colour, rows=0 ):
        print('Rows: ', rows)
        print('Image: ', self.image.shape  )


        self.image, new_seams = sc.add_rows(self.image, energy_function, rows)
        print('new_seams: ', new_seams.shape)
        self.added_seams.extend(new_seams)
        self.added_order.extend([True for _ in range(rows)])

    # Always add new cols to the image, needed for the removal
    def add_always_new_cols(self, energy_function=sc.e1_colour, cols=0 ):
        self.image, new_seams = sc.add_columns(self.image, energy_function, cols)
        self.added_seams.extend(new_seams)
        self.added_order.extend([False for _ in range(cols)])

    def always_add(self, energy_function=sc.e1_colour, rows=0, cols=0):
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

if __name__ == '__main__':
    im = cv2.cvtColor(cv2.imread(".\\Figures\\Castle.jpg"), cv2.COLOR_BGR2RGB)
    my_seam_image = SeamImage(im)

    start = time.time()
    my_seam_image.add_rows_and_cols(rows=75, cols=150)
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



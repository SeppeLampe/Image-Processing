import os
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog

import cv2
import imutils
import numpy as np
from PIL import ImageTk, Image, ImageDraw
from SeamImage import *

DEFAULT_IMAGE = 'Figures\\Castle.jpg'
DEFAULT_SIZE = 400


class SeamCarvingGUI(tk.Frame):
    """
    Class for the Seam Carving GUI layout implemented with Tkinter, the Python binding to the Tk GUI toolkit
    """

    def __init__(self, master):
        """
        Constructor initializes class attributes and GUI layout
        :param master: parent widget and it is an optional parameter
        """
        super(SeamCarvingGUI, self).__init__(master)
        self.pack()
        self.master = master

        """ Class attributes """
        # For drawing
        self.last_x, self.last_y = None, None
        self.color = 'red'
        self.color_code = (255, 0, 0)

        # For displaying images
        self.seam_image = None
        # The ghost_image is not visible and it is used to retrieve the mask for eventual object removal functionality
        self.ghost_image = Image.new("RGB", (DEFAULT_SIZE, DEFAULT_SIZE), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.ghost_image)

        """ GUI bound variables """
        self.file_path = tk.StringVar()
        self.width_var = tk.IntVar()
        self.height_var = tk.IntVar()
        self.keep_shape_var = tk.IntVar()
        self.factor_var = tk.DoubleVar(value=1.2)
        self.output_img_name = tk.StringVar(value='result')

        # Initialize GUI elements
        self.initialize_gui()

    def initialize_gui(self):
        """
        Method called from the constructor, segmented for better readability.
        Contains the initialization of the whole GUI window layout.
        The layout is split into two main parts:

         |--------------|
         |  TOP FRAME   |
         |--------------|
         | BOTTOM FRAME |
         |--------------|

         The 'top frame' contains multiple functional components for interaction.
         The 'bottom frame' contains the image display.
         :return: Nothing
        """
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, expand=True)

        # First section in the top frame contains components for selecting the image file
        file_frame = tk.Frame(top_frame)
        file_frame.pack(side=tk.TOP, expand=True, fill=tk.X)

        browse_button = tk.Button(file_frame, text="Browse...", anchor='w', command=self.add_image)
        browse_button.pack(side=tk.LEFT)

        label = tk.Label(file_frame, anchor='w', textvariable=self.file_path)
        label.pack(side=tk.LEFT)

        # Separator lines are used between sections for better visibility
        separator = ttk.Separator(top_frame, orient='horizontal')
        separator.pack(fill=tk.X)

        # Second section in the top frame contains elements related to resizing the image
        resize_frame = tk.Frame(top_frame)
        resize_frame.pack(side=tk.TOP, expand=True, fill=tk.X)

        label = tk.Label(resize_frame, text='Resize image', width=15, anchor='w', font='Helvetica 14 bold')
        label.pack(side=tk.LEFT)

        width_label = tk.Label(resize_frame, text="Width:")
        width_label.pack(side=tk.LEFT)

        width_entry = tk.Entry(resize_frame, textvariable=self.width_var)
        width_entry.pack(side=tk.LEFT)

        height_label = tk.Label(resize_frame, text="Height:")
        height_label.pack(side=tk.LEFT)

        height_entry = tk.Entry(resize_frame, textvariable=self.height_var)
        height_entry.pack(side=tk.LEFT)

        execute_resize_button = tk.Button(resize_frame, text="Execute", command=self.resize_image)
        execute_resize_button.pack(side=tk.LEFT)

        reset_resize_button = tk.Button(resize_frame, text="Reset", command=self.reset_img)
        reset_resize_button.pack(side=tk.LEFT)

        # Separator lines are used between sections for better visibility
        separator = ttk.Separator(top_frame, orient='horizontal')
        separator.pack(fill=tk.X)

        # Third section in the top frame contains elements related to object removal
        obj_remove_frame = tk.Frame(top_frame)
        obj_remove_frame.pack(side=tk.TOP, expand=True, fill=tk.X)

        label = tk.Label(obj_remove_frame, text='Remove object', width=15, anchor='w', font='Helvetica 14 bold')
        label.pack(side=tk.LEFT)

        red_button = tk.Button(obj_remove_frame, text="RED", fg="red", command=self.select_red)
        red_button.pack(side=tk.LEFT)

        green_button = tk.Button(obj_remove_frame, text="GREEN", fg="green", command=self.select_green)
        green_button.pack(side=tk.LEFT)

        keep_shape_checkbox = tk.Checkbutton(obj_remove_frame, text="Keep shape?", variable=self.keep_shape_var)
        keep_shape_checkbox.pack(side=tk.LEFT)

        execute_obj_remove_button = tk.Button(obj_remove_frame, text="Execute", command=self.remove_object_image)
        execute_obj_remove_button.pack(side=tk.LEFT)

        reset_remove_obj_button = tk.Button(obj_remove_frame, text="Reset", command=self.reset_img)
        reset_remove_obj_button.pack(side=tk.LEFT)

        # Separator lines are used between sections for better visibility
        separator = ttk.Separator(top_frame, orient='horizontal')
        separator.pack(fill=tk.X)

        # Fourth section in the top frame contains elements related to content amplification
        content_ampl_frame = tk.Frame(top_frame)
        content_ampl_frame.pack(side=tk.TOP, expand=True, fill=tk.X)

        label = tk.Label(content_ampl_frame, text='Amplify content', width=15, anchor='w', font='Helvetica 14 bold')
        label.pack(side=tk.LEFT)

        factor_label = tk.Label(content_ampl_frame, text="Amplification factor:")
        factor_label.pack(side=tk.LEFT)

        factor_entry = tk.Entry(content_ampl_frame, textvariable=self.factor_var)
        factor_entry.pack(side=tk.LEFT)

        execute_amplify_button = tk.Button(content_ampl_frame, text="Execute", command=self.amplify_content)
        execute_amplify_button.pack(side=tk.LEFT)

        reset_amplify_button = tk.Button(content_ampl_frame, text="Reset", command=self.reset_img)
        reset_amplify_button.pack(side=tk.LEFT)

        # Separator lines are used between sections for better visibility
        separator = ttk.Separator(top_frame, orient='horizontal')
        separator.pack(fill=tk.X)

        # Fourth section in the top frame contains elements related to content amplification
        save_frame = tk.Frame(top_frame)
        save_frame.pack(side=tk.TOP, expand=True, fill=tk.X)

        output_label = tk.Label(save_frame, text="Output image name:")
        output_label.pack(side=tk.LEFT)

        output_entry = tk.Entry(save_frame, textvariable=self.output_img_name)
        output_entry.pack(side=tk.LEFT)

        save_button = tk.Button(save_frame, text="Save result", command=self.save_image)
        save_button.pack(side=tk.LEFT)

        # Separator lines are used between sections for better visibility
        separator = ttk.Separator(top_frame, orient='horizontal')
        separator.pack(fill=tk.X)

        # Bottom frame contains a canvas for displaying the image and drawing
        canvas_frame = tk.Frame(top_frame)
        canvas_frame.pack(side=tk.TOP, expand=True)

        self.canvas = tk.Canvas(canvas_frame, width=DEFAULT_SIZE, height=DEFAULT_SIZE, bg='black')
        self.canvas.pack(fill=tk.BOTH)

        self.image = ImageTk.PhotoImage(Image.fromarray(np.asarray([0, 0, 0])))
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.image, anchor='nw')

        # Bind drawing functionalities to mouse left click and mouse left button hold
        self.canvas.bind("<Button-1>", self.get_x_and_y)
        self.canvas.bind("<B1-Motion>", self.draw_on_image)

    def add_image(self):
        """
        This method solves selecting an image file and displaying it in the GUI
        :return: Nothing
        """
        # If there was previous drawing on the canvas, remove it
        self.canvas.delete("line")

        # Prompts the user for file selection via a file dialog
        img_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Open Image",
                                              filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        self.file_path.set(img_path)
        self.seam_image = SeamImage(img_path)

        # Adjust GUI to selected image
        self.post_process()

    def reset_img(self):
        """
        Reset functionality discards any modification done to the original image
        :return: Nothing
        """
        # Delete any drawing from the canvas
        self.canvas.delete("line")

        if self.seam_image is not None:
            self.seam_image.reset()
            # Update display
            self.post_process()

    def post_process(self):
        """
        This method adjusts the GUI elements to any resizing change that was done to the displayed image
        :return: Nothing
        """
        # Get the height and weight and we don't care about the number of channels
        height, width, _ = self.seam_image.get_image().shape

        # Set the width and height variables to display the updated values
        self.width_var.set(width)
        self.height_var.set(height)

        # Adjust the canvas and the newly displayed image element accordingly
        self.canvas.config(width=width, height=height)
        new_image = self.seam_image.get_image().astype(np.uint8)  # Convert to 8 bit unsigned integer for display
        self.image = ImageTk.PhotoImage(Image.fromarray(new_image))
        self.canvas.itemconfig(self.image_on_canvas, image=self.image, anchor='nw')

        # The ghost_image is used to retrieve the mask, it has to be the same size as the image
        self.ghost_image = self.ghost_image.resize((width, height))
        self.draw = ImageDraw.Draw(self.ghost_image)

    def select_red(self):
        """
        Set drawing color to red
        :return: Nothing
        """
        self.color = 'red'
        self.color_code = (255, 0, 0)

    def select_green(self):
        """
        Set drawing color to green
        :return: Nothing
        """
        self.color = 'green'
        self.color_code = (0, 255, 0)

    def get_x_and_y(self, event):
        """
        Update the last registered coordinates of a mouse event
        :param event: I/O event, in out case a mouse click
        :return: Nothing
        """
        self.last_x, self.last_y = event.x, event.y

    def draw_on_image(self, event):
        """
        Draw onto the image canvas
        :param event: I/O event, in out case a mouse drag
        :return: Nothing
        """
        # Our drawing will be simulated by repeatedly drawing a line
        # between the previous mouse location and the current one
        self.canvas.create_line((self.last_x, self.last_y, event.x, event.y), fill=self.color, width=10, tag="line")
        # Draw on the invisible mask image as well
        self.draw.line([self.last_x, self.last_y, event.x, event.y], width=10, fill=self.color_code)
        # Update coordinates of the last event
        self.last_x, self.last_y = event.x, event.y

    def resize_image(self):
        """
        Method for executing the resize via seam carving functionality
        :return: Nothing
        """
        # Check for proper numeric input and warn the user otherwise
        try:
            # Retrieve input width and height
            target_width = self.width_var.get()
            target_height = self.height_var.get()

            # Check for positive input and warn the user otherwise
            if target_height <= 0:
                messagebox.showerror("Error!", "Please input a positive height value!")
                return

            if target_width <= 0:
                messagebox.showerror("Error!", "Please input a positive width value!")
                return

            # Call the seam carving function
            self.seam_image.resize(height=target_height, width=target_width)
            # Update display
            self.post_process()
        except tk.TclError:
            messagebox.showerror("Error!", "Width and height must be integers!")

    def remove_object_image(self):
        """
        Use seam carving to remove objects from the image
        :return: Nothing
        """
        # The ghost image contains the drawn red and green areas and black in the rest
        color_mask = self.ghost_image

        # Retrieve the red areas for removal and the green areas for keeping
        r, g, b = cv2.split(np.array(color_mask))
        red_mask = np.where(r, 1, 0)
        green_mask = np.where(g, 1, 0)

        keep_shape = True if self.keep_shape_var.get() else False
        self.seam_image.remove_mask(mask=red_mask, keep_shape=keep_shape)

        # Remove all drawings
        self.canvas.delete("line")
        # Update display
        self.post_process()

    def amplify_content(self):
        """
        Amplify content via seam carving
        :return: Nothing
        """
        # Check for proper numeric input and warn the user otherwise
        try:
            # Retrieve input factor
            target_factor = self.factor_var.get()

            # Check for positive input and warn the user otherwise
            if target_factor < 1:
                messagebox.showerror("Error!", "Please input a value greater than 1!")
                return

            # Call the seam carving function
            self.seam_image.amplify_content(target_factor)
            # Update display
            self.post_process()
        except tk.TclError:
            messagebox.showerror("Error!", "Factor must be numeric value!")

        # Update display
        self.post_process()

    def save_image(self):
        """
        Save result image on disk
        :return: Nothing
        """
        output_file_name = self.output_img_name.get() + '.png'
        image = self.seam_image.get_image().astype(np.uint8)
        # The returned image is in BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_file_name, image)


if __name__ == '__main__':
    # Initialize main window
    root = tk.Tk()
    root.title('Seam Carving')
    SeamCarvingGUI(master=root)

    # This method starts up the tkinter GUI
    root.mainloop()

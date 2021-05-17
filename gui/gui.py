import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import cv2
import imutils
import numpy as np
from PIL import ImageTk, Image, ImageDraw

DEFAULT_IMAGE = '..\\Figures\\Castle.jpg'


class SeamCarvingGUI(tk.Frame):
    def __init__(self, master):
        super(SeamCarvingGUI, self).__init__(master)
        self.pack()
        self.master = master

        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, expand=True)

        file_frame = tk.Frame(top_frame)
        file_frame.pack(side=tk.TOP, expand=True, fill=tk.X)

        browse_button = tk.Button(file_frame, text="Browse...", anchor='w', command=self.add_image)
        browse_button.pack(side=tk.LEFT)

        self.file_path = tk.StringVar()
        label = tk.Label(file_frame, anchor='w', textvariable=self.file_path)
        label.pack(side=tk.LEFT)

        separator1 = ttk.Separator(top_frame, orient='horizontal')
        separator1.pack(fill=tk.X)

        resize_frame = tk.Frame(top_frame)
        resize_frame.pack(side=tk.TOP, expand=True, fill=tk.X)

        label = tk.Label(resize_frame, text='Resize', width=8, anchor='w', font='Helvetica 18 bold')
        label.pack(side=tk.LEFT)

        self.width_var = tk.IntVar()
        width_label = tk.Label(resize_frame, text="Width:")
        width_entry = tk.Entry(resize_frame, textvariable=self.width_var)
        width_label.pack(side=tk.LEFT)
        width_entry.pack(side=tk.LEFT)

        self.height_var = tk.IntVar()
        height_label = tk.Label(resize_frame, text="Height:")
        height_entry = tk.Entry(resize_frame, textvariable=self.height_var)
        height_label.pack(side=tk.LEFT)
        height_entry.pack(side=tk.LEFT)

        execute_resize_button = tk.Button(resize_frame, text="Execute", command=self.resize_image)
        execute_resize_button.pack(side=tk.LEFT)

        reset_resize_button = tk.Button(resize_frame, text="Reset", command=self.reset_img)
        reset_resize_button.pack(side=tk.LEFT)

        separator2 = ttk.Separator(top_frame, orient='horizontal')
        separator2.pack(fill=tk.X)

        retarget_frame = tk.Frame(top_frame)
        retarget_frame.pack(side=tk.TOP, expand=True, fill=tk.X)

        label = tk.Label(retarget_frame, text='Retarget', width=8, anchor='w', font='Helvetica 18 bold')
        label.pack(side=tk.LEFT)

        red_button = tk.Button(retarget_frame, text="RED", fg="red", command=self.select_red)
        red_button.pack(side=tk.LEFT)

        red_button = tk.Button(retarget_frame, text="GREEN", fg="green", command=self.select_green)
        red_button.pack(side=tk.LEFT)

        execute_retarget_button = tk.Button(retarget_frame, text="Execute", command=self.retarget_image)
        execute_retarget_button.pack(side=tk.LEFT)

        reset_retarget_button = tk.Button(retarget_frame, text="Reset", command=self.reset_img)
        reset_retarget_button.pack(side=tk.LEFT)

        separator3 = ttk.Separator(top_frame, orient='horizontal')
        separator3.pack(fill=tk.X)

        canvas_frame = tk.Frame(top_frame)
        canvas_frame.pack(side=tk.TOP, expand=True)

        self.canvas = tk.Canvas(canvas_frame, width=400, height=400, bg='black')
        self.canvas.pack(fill=tk.BOTH)

        self.image = ImageTk.PhotoImage(Image.fromarray(np.asarray([0, 0, 0])))
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.image, anchor='nw')

        self.canvas.bind("<Button-1>", self.get_x_and_y)
        self.canvas.bind("<B1-Motion>", self.draw_smth)

        self.lasx, self.lasy = None, None
        self.orig_img = None
        self.color = 'red'
        self.color_code = (255, 0, 0)

        self.ghost_image = Image.new("RGB", (400, 400), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.ghost_image)

    def add_image(self):
        img_path = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), '..'), title="Open Image",
                                              filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        self.file_path.set(img_path)
        cv_img = cv2.imread(img_path)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, _ = cv_img.shape
        if width < height:
            cv_img = imutils.resize(cv_img, width=400)
        else:
            cv_img = imutils.resize(cv_img, height=400)
        height, width, _ = cv_img.shape
        self.width_var.set(width)
        self.height_var.set(height)
        self.orig_img = cv_img
        self.ghost_image = self.ghost_image.resize((width, height))
        self.draw = ImageDraw.Draw(self.ghost_image)
        self.image = ImageTk.PhotoImage(Image.fromarray(cv_img))
        self.canvas.config(width=width, height=height)
        self.canvas.itemconfig(self.image_on_canvas, image=self.image, anchor='nw')

    def reset_img(self):
        self.canvas.delete("line")
        if self.orig_img is not None:
            self.image = ImageTk.PhotoImage(Image.fromarray(self.orig_img))
            self.canvas.itemconfig(self.image_on_canvas, image=self.image, anchor='nw')
            height, width, _ = self.orig_img.shape
            self.width_var.set(width)
            self.height_var.set(height)

    def select_red(self):
        self.color = 'red'
        self.color_code = (255, 0, 0)

    def select_green(self):
        self.color = 'green'
        self.color_code = (0, 255, 0)

    def get_x_and_y(self, event):
        self.lasx, self.lasy = event.x, event.y

    def draw_smth(self, event):
        self.canvas.create_line((self.lasx, self.lasy, event.x, event.y), fill=self.color, width=10, tag="line")
        self.draw.line([self.lasx, self.lasy, event.x, event.y], width=10, fill=self.color_code)
        self.lasx, self.lasy = event.x, event.y

    def resize_image(self):
        target_width = self.width_var.get()
        target_height = self.height_var.get()
        print('Width: {0} Height: {1}'.format(target_width, target_height))
        # TODO do something with these and return a cv2 image
        # self.image = ImageTk.PhotoImage(Image.fromarray(<RESULT HERE>))
        # self.canvas.itemconfig(self.image_on_canvas, image=self.image, anchor='nw')

    def retarget_image(self):
        mask = self.ghost_image
        r, g, b = cv2.split(np.array(mask))
        red_mask = np.where(r, 1, 0)
        green_mask = np.where(g, 1, 0)
        # TODO do something with these and return a cv2 image
        # self.image = ImageTk.PhotoImage(Image.fromarray(<RESULT HERE>))
        # self.canvas.itemconfig(self.image_on_canvas, image=self.image, anchor='nw')


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Seam Carving')
    SeamCarvingGUI(master=root)
    root.mainloop()

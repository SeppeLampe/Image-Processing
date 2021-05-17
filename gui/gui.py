import os
import tkinter as tk
from tkinter import filedialog

import cv2
import imutils
import numpy as np
from PIL import ImageTk, Image

DEFAULT_IMAGE = '..\\Figures\\Castle.jpg'


class SeamCarvingGUI(tk.Frame):
    def __init__(self, master):
        super(SeamCarvingGUI, self).__init__(master)
        self.pack()
        self.master = master

        self.frame = tk.Frame(self)
        self.frame.pack(side=tk.TOP, expand=True, fill=tk.X)

        self.button = tk.Button(self.frame, text="OPEN", command=self.add_image)
        self.button.pack(side=tk.LEFT)

        self.file_path = tk.StringVar()
        self.label = tk.Label(self.frame, textvariable=self.file_path)
        self.label.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self, width=400, height=400, bg='black')
        self.canvas.pack(fill=tk.BOTH)

        self.image = ImageTk.PhotoImage(Image.fromarray(np.asarray([0, 0, 0])))
        self.image_on_canvas = self.canvas.create_image(0, 0, image=self.image, anchor='nw')

        self.canvas.bind("<Button-1>", self.get_x_and_y)
        self.canvas.bind("<B1-Motion>", self.draw_smth)

        self.lasx, self.lasy = None, None

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
        self.image = ImageTk.PhotoImage(Image.fromarray(cv_img))
        self.canvas.config(width=width, height=height)
        self.canvas.itemconfig(self.image_on_canvas, image=self.image, anchor='nw')

    def get_x_and_y(self, event):
        self.lasx, self.lasy = event.x, event.y

    def draw_smth(self, event):
        self.canvas.create_line((self.lasx, self.lasy, event.x, event.y), fill='red', width=2)
        self.lasx, self.lasy = event.x, event.y


if __name__ == '__main__':
    root = tk.Tk()
    SeamCarvingGUI(master=root)
    root.mainloop()

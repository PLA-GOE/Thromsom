import os
import tkinter as tk
from pathlib import Path
from tkinter import NW, messagebox
from tkinter.filedialog import askopenfilename

# Directory Dialog
import numpy as np
import pydicom
from PIL import Image, ImageTk
from pydicom.filereader import dcmread

from obj import image_panel

dataset = None
global_cut_params = None

class draw_gui(tk.Toplevel):
    def motion(self, event):
        if self.line_counter == 0:
            self.draw_canvas.delete(self.line_x1)
            self.line_x1 = self.draw_canvas.create_line(0, event.y, 1024, event.y, fill="#ff0000")
        elif self.line_counter == 1:
            self.draw_canvas.delete(self.line_x2)
            self.line_x2 = self.draw_canvas.create_line(0, event.y, 1024, event.y, fill="#ff0000")
        elif self.line_counter == 2:
            self.draw_canvas.delete(self.line_y1)
            self.line_y1 = self.draw_canvas.create_line(event.x, 0, event.x, 1024, fill="#00ff00")
        elif self.line_counter == 3:
            self.draw_canvas.delete(self.line_y2)
            self.line_y2 = self.draw_canvas.create_line(event.x, 0, event.x, 1024, fill="#00ff00")
        elif self.line_counter == 4:
            self.draw_canvas.delete(self.line_x_1)
            self.line_x_1 = self.draw_canvas.create_line(event.x-5, event.y-5, event.x+6, event.y+6, fill="#0000ff")
            self.draw_canvas.delete(self.line_x_2)
            self.line_x_2 = self.draw_canvas.create_line(event.x+6, event.y-5, event.x-5, event.y+6, fill="#0000ff")

    def click(self, event):
        print("clicked at", event.x, event.y, self.line_counter)
        self.draw_canvas.create_image(0, 0, anchor=NW, image=self.img)
        if self.line_counter == 0:
            self.line_vert.append(event.y)
            self.line_x1 = self.draw_canvas.create_line(0, event.y, 1024, event.y, fill="#ff0000")
        if self.line_counter == 1:
            self.line_x1 = self.draw_canvas.create_line(0, self.line_vert[0], 1024, self.line_vert[0], fill="#ff0000")
            self.line_x2 = self.draw_canvas.create_line(0, event.y, 1024, event.y, fill="#ff0000")
            self.line_vert.append(event.y)
        if self.line_counter == 2:
            self.line_x1 = self.draw_canvas.create_line(0, self.line_vert[0], 1024, self.line_vert[0], fill="#ff0000")
            self.line_x2 = self.draw_canvas.create_line(0, self.line_vert[1], 1024, self.line_vert[1], fill="#ff0000")
            self.line_y1 = self.draw_canvas.create_line(event.x, 0, event.x, 1024, fill="#00ff00")
            self.line_hor.append(event.x)
        if self.line_counter == 3:
            self.line_x1 = self.draw_canvas.create_line(0, self.line_vert[0], 1024, self.line_vert[0], fill="#ff0000")
            self.line_x2 = self.draw_canvas.create_line(0, self.line_vert[1], 1024, self.line_vert[1], fill="#ff0000")
            self.line_y1 = self.draw_canvas.create_line(self.line_hor[0], 0, self.line_hor[0], 1024, fill="#00ff00")
            self.line_y2 = self.draw_canvas.create_line(event.x, 0, event.x, 1024, fill="#00ff00")
            self.line_hor.append(event.x)
        if self.line_counter == 4:
            self.line_x1 = self.draw_canvas.create_line(0, self.line_vert[0], 1024, self.line_vert[0], fill="#ff0000")
            self.line_x2 = self.draw_canvas.create_line(0, self.line_vert[1], 1024, self.line_vert[1], fill="#ff0000")
            self.line_y1 = self.draw_canvas.create_line(self.line_hor[0], 0, self.line_hor[0], 1024, fill="#00ff00")
            self.line_y2 = self.draw_canvas.create_line(self.line_hor[1], 0, self.line_hor[1], 1024, fill="#00ff00")
            self.line_x_1 = self.draw_canvas.create_line(event.x-5, event.y-5, event.x+6, event.y+6, fill="#0000ff")
            self.line_x_2 = self.draw_canvas.create_line(event.x+6, event.y-5, event.x-5, event.y+6, fill="#0000ff")
            self.center_point = (event.x, event.y)
            answer = messagebox.askyesno(title="Bestätigung des Areals", message="Ist das Areal korrekt abgesteckt?")
            if answer:
                print("Start")
                self.line_counter += 2
                global global_cut_params
                image_panel.generate_images(self.line_vert, self.line_hor, self.center_point, self.image_array.shape, self.image_array)
            else:
                if self.line_counter <= 3:
                    self.line_counter += 1
                    self.unclick(event)
        if self.line_counter <= 3:
            self.line_counter += 1

    def unclick(self, event):
        print("unclicked at", event.x, event.y, self.line_counter)
        self.draw_canvas.create_image(0, 0, anchor=NW, image=self.img)
        if self.line_counter == 1:
            self.line_vert.pop()
            self.line_counter -= 1
        elif self.line_counter == 2:
            self.line_vert.pop()
            self.line_x1 = self.draw_canvas.create_line(0, self.line_vert[0], 1024, self.line_vert[0], fill="#ff0000")
            self.line_counter -= 1
        elif self.line_counter == 3:
            self.line_hor.pop()
            self.line_x1 = self.draw_canvas.create_line(0, self.line_vert[0], 1024, self.line_vert[0], fill="#ff0000")
            self.line_x2 = self.draw_canvas.create_line(0, self.line_vert[1], 1024, self.line_vert[1], fill="#ff0000")
            self.line_counter -= 1
        elif self.line_counter == 4:
            self.line_hor.pop()
            self.line_x1 = self.draw_canvas.create_line(0, self.line_vert[0], 1024, self.line_vert[0], fill="#ff0000")
            self.line_x2 = self.draw_canvas.create_line(0, self.line_vert[1], 1024, self.line_vert[1], fill="#ff0000")
            self.line_y1 = self.draw_canvas.create_line(self.line_hor[0], 0, self.line_hor[0], 1024, fill="#00ff00")
            self.line_counter -= 1
        elif self.line_counter == 5:
            self.center_point = (-1, -1)
            self.line_x1 = self.draw_canvas.create_line(0, self.line_vert[0], 1024, self.line_vert[0], fill="#ff0000")
            self.line_x2 = self.draw_canvas.create_line(0, self.line_vert[1], 1024, self.line_vert[1], fill="#ff0000")
            self.line_y1 = self.draw_canvas.create_line(self.line_hor[0], 0, self.line_hor[0], 1024, fill="#00ff00")
            self.line_y2 = self.draw_canvas.create_line(self.line_hor[1], 0, self.line_hor[1], 1024, fill="#00ff00")
            self.line_counter -= 1
        self.motion(event)

    def __init__(self, path, *args, **kwargs):
        self.image_array = None
        self.line_x1 = None
        self.line_x2 = None
        self.line_y1 = None
        self.line_y2 = None
        self.line_x_1 = None
        self.line_x_2 = None
        self.line_counter = 0
        self.line_vert = []
        self.line_hor = []
        self.center_point = (-1,-1)
        tk.Toplevel.__init__(self, *args, **kwargs)
        self.ifa = None
        self.img = self.load_dcm(path)
        self.draw_canvas = None
        print(self.ifa)
        print(self.img)

        # Define the geometry of the window
        self.geometry("1024x1024")
        self.draw_canvas = tk.Canvas(self, width=1024, height=1024)
        self.draw_canvas.create_image(0, 0, anchor=NW, image=self.img)
        self.draw_canvas.bind("<Button-1>", self.click)
        self.draw_canvas.bind("<Button-3>", self.unclick)
        self.draw_canvas.bind("<Motion>", self.motion)
        self.draw_canvas.pack()
        # Create a Label Widget to display the text or Image

        self.mainloop()

    def load_dcm(self, path):
        global dataset
        ds = dcmread(str(path))
        dataset = ds
        root_dir = Path(ds.filename).resolve().parent
        print(ds)
        print(ds.pixel_array.shape)
        self.image_array = ds.pixel_array
        # Average of first and last slice
        min_arr = np.zeros((2,ds.pixel_array.shape[1],ds.pixel_array.shape[2]))
        min_arr[0] = ds.pixel_array[0]
        min_arr[1] = ds.pixel_array[-1]
        print(min_arr.shape)
        click_slice = np.amin(min_arr, axis=0)
        print(click_slice.shape)
        fq_perc = int(np.percentile(click_slice, 75))
        tq_perc = int(np.percentile(click_slice, 90))
        print("First:", fq_perc, ", third:", tq_perc)
        click_slice[click_slice < fq_perc] = fq_perc
        click_slice[click_slice > tq_perc] = tq_perc
        print(np.amax(click_slice))
        print(np.amin(click_slice))
        image_raw = (((click_slice - np.amin(click_slice)) / (np.amax(click_slice) - np.amin(click_slice))) * 255.0).astype("uint8")
        print(image_raw.shape)
        print(image_raw)
        self.ifa = Image.fromarray(image_raw)
        self.ifa = self.ifa.resize((1024, 1024), Image.ANTIALIAS)
        return ImageTk.PhotoImage(master=self, image=self.ifa)

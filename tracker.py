import copy
import os
import time
from datetime import datetime
import tkinter as tk
from tkinter import W, filedialog, simpledialog, DISABLED
import nibabel as nb

from pathlib import Path

import math
import numpy as np

from PIL import Image, ImageTk
from pydicom import dcmread

from analyze import collapse, stats, find_id, analyze_id, find_area
from obj.helpers import CreateToolTip
from obj.global_region import global_region
from obj.region import Region, translate_color_id, translate_color_id_to_rgb

img = None
index_label = tk.Label
color_id_sv = tk.StringVar
color_id = 0
cst = "0.8"
global_region_scan = None
init_dir = "/"


def show_mono(image_in):
    image_min = np.amin(image_in)
    show_image = (((image_in - image_min) / (np.amax(image_in) - image_min)) * 255.0).astype(np.uint8)
    img = Image.fromarray(show_image, 'L')
    img.show()


def show_color(image_in):
    image_min = np.amin(image_in)
    show_image = (((image_in - image_min) / (np.amax(image_in) - image_min)) * 255.0).astype(np.uint8)
    # show_image = np.rot90(show_image)
    img = Image.fromarray(show_image, 'RGB')
    img.show()

def increase_cid():
    global canvas
    global color_id
    global color_id_sv
    if color_id < 9:
        color_id += 1
    color_id_sv.set(str(color_id))
    index_label.config(fg=translate_color_id(color_id))
    update_canvas()
    print(color_id, "---", color_id_sv.get())


def decrease_cid():
    global canvas
    global color_id
    global color_id_sv

    if color_id > 0:
        color_id -= 1
    color_id_sv.set(str(color_id))
    index_label.config(fg=translate_color_id(color_id))
    update_canvas()
    print(color_id, "---", color_id_sv.get())


def key_press(event, global_region_scan):
    global img
    global color_id
    if event.keysym == "Right":
        increase_cid()
    if event.keysym == "Left":
        decrease_cid()
    if event.keysym == "Up":
        global_region_scan.increase_slice(1)
        img = ImageTk.PhotoImage(image=Image.fromarray(np.fliplr(np.rot90(global_region_scan.get_current_slice(), axes=(1, 0)))))
        update_canvas()
    if event.keysym == "Down":
        global_region_scan.increase_slice(-1)
        img = ImageTk.PhotoImage(image=Image.fromarray(np.fliplr(np.rot90(global_region_scan.get_current_slice(), axes=(1, 0)))))
        update_canvas()

def z_trace(global_region_scan):
    print("Z-Trace")
    for region in global_region_scan.get_current_regions():
        print("Working region:", region.get_color_id())
        color_id = region.get_color_id()
        start_index = global_region_scan.get_current_slice_index()
        region_points = region.get_visited_pixel_list()
        print("Found:", len(region_points), "visited Pixels")
        start_index -= 1
        while start_index > 0:
            for pixel in region_points:
                if region.get_lower_thresh() < global_region_scan.get_voxel_from_cube(pixel[0], pixel[1], start_index) < region.get_upper_thresh():
                    slice_click(pixel[0], pixel[1], start_index, global_region_scan, color_id)
            if global_region_scan.get_color_id_region(start_index, color_id):
                global_region_scan.get_color_id_region(start_index, color_id).track()
            start_index -= 1
        start_index = global_region_scan.get_current_slice_index()
        start_index += 1
        while start_index < global_region_scan.get_z():
            for pixel in region_points:
                if region.get_lower_thresh() < global_region_scan.get_voxel_from_cube(pixel[0], pixel[1], start_index) < region.get_upper_thresh():
                    slice_click(pixel[0], pixel[1], start_index, global_region_scan, color_id)
            if global_region_scan.get_color_id_region(start_index, color_id):
                global_region_scan.get_color_id_region(start_index, color_id).track()
            start_index += 1
    update_canvas()


def long_trace(global_region_scan):
    print("Long-Trace")
    global color_id
    for region in global_region_scan.get_current_regions():
        print("Working region:", region.get_color_id())
        color_id = region.get_color_id()
        start_index = global_region_scan.get_current_slice_index()
        region_points = region.get_visited_pixel_list()
        print("Found:", len(region_points), "visited Pixels")
        start_index -= 1
        while start_index >= 0:
            for pixel in region_points:
                if region.get_lower_thresh() <= global_region_scan.get_voxel_from_cube(pixel[0], pixel[1], start_index) <= region.get_upper_thresh():
                    slice_click(pixel[0], pixel[1], start_index, global_region_scan, color_id)
            if global_region_scan.get_color_id_region(start_index, color_id):
                global_region_scan.get_color_id_region(start_index, color_id).set_lower_thresh(region.get_lower_thresh())
                global_region_scan.get_color_id_region(start_index, color_id).set_upper_thresh(region.get_upper_thresh())
                global_region_scan.get_color_id_region(start_index, color_id).track()
                region_points = global_region_scan.get_color_id_region(start_index, color_id).get_visited_pixel_list()
            start_index -= 1
        start_index = global_region_scan.get_current_slice_index()
        start_index += 1
        while start_index < global_region_scan.get_z():
            for pixel in region_points:
                if region.get_lower_thresh() <= global_region_scan.get_voxel_from_cube(pixel[0], pixel[1], start_index) <= region.get_upper_thresh():
                    slice_click(pixel[0], pixel[1], start_index, global_region_scan, color_id)
            if global_region_scan.get_color_id_region(start_index, color_id):
                global_region_scan.get_color_id_region(start_index, color_id).set_lower_thresh(region.get_lower_thresh())
                global_region_scan.get_color_id_region(start_index, color_id).set_upper_thresh(region.get_upper_thresh())
                global_region_scan.get_color_id_region(start_index, color_id).track()
                region_points = global_region_scan.get_color_id_region(start_index, color_id).get_visited_pixel_list()
            start_index += 1
    update_canvas()

def slice_trace(global_region_scan):
    global color_id
    global_region_scan.get_current_color_id_region(color_id).track_region_blocked(global_region_scan.get_all_visited_pixels())
    update_canvas()

def all_slice_trace(global_region_scan):
    global color_id
    th_lo = global_region_scan.get_current_color_id_region(color_id).get_lower_thresh()
    th_up = global_region_scan.get_current_color_id_region(color_id).get_upper_thresh()
    for z_index in range(global_region_scan.get_z()):
        #print(global_region_scan.get_all_visited_pixels_z(z_index))
        #print(np.where(global_region_scan.get_all_visited_pixels_z(z_index) != 0))
        #print(global_region_scan.get_slice(z_index))
        #print(np.where((th_lo <= global_region_scan.get_slice(z_index))& (global_region_scan.get_slice(z_index)<= th_up)))
        no_visit_array = np.where((global_region_scan.get_all_visited_pixels_z(z_index) != 0)&((th_lo <= global_region_scan.get_slice(z_index))& (global_region_scan.get_slice(z_index)<= th_up)))
        for pixel_index in range(len(no_visit_array[0])):
            print(pixel_index,":",no_visit_array[0][pixel_index],"-", no_visit_array[1][pixel_index],"-", z_index)
            slice_click(no_visit_array[0][pixel_index], no_visit_array[1][pixel_index], z_index, global_region_scan, color_id)
        if global_region_scan.get_color_id_region(z_index, color_id):
            global_region_scan.get_color_id_region(z_index, color_id).track_region_blocked(global_region_scan.get_all_visited_pixels_z(z_index))
    update_canvas()


def near_trace(global_region_scan):
    print("Near-Trace")
    for region in global_region_scan.get_current_regions():
        print("Working region:", region.get_color_id())
        color_id_loc = region.get_color_id()
        start_index = global_region_scan.get_current_slice_index()
        region_points = region.get_visited_pixel_list()
        print("Found:", len(region_points), "visited Pixels")
        start_index -= 1
        while start_index >= 0:
            if global_region_scan.get_color_id_region(start_index, color_id_loc):
                global_region_scan.delete_color_region(start_index, color_id_loc)
            for pixel in region_points:
                if region.get_lower_thresh() <= global_region_scan.get_voxel_from_cube(pixel[0], pixel[1], start_index) <= region.get_upper_thresh():
                    slice_click(pixel[0], pixel[1], start_index, global_region_scan, color_id_loc)
            similar_colored_voxels = global_region_scan.get_threshed_slice(start_index, region.get_upper_thresh(), region.get_lower_thresh())
            min_dist = 4096
            min_index = -1
            for pixel_index in range(len(region_points)):
                for sim_index in range(len(similar_colored_voxels[0])):
                    distance = math.sqrt((region_points[pixel_index][0]-similar_colored_voxels[0][sim_index])**2+(region_points[pixel_index][1]-similar_colored_voxels[1][sim_index])**2)
                    if distance < min_dist:
                        min_dist = distance
                        min_index = sim_index
                    if min_dist == 0:
                        break
            if min_index != -1:
                slice_click(similar_colored_voxels[0][min_index], similar_colored_voxels[1][min_index], start_index, global_region_scan, color_id_loc)
                global_region_scan.get_color_id_region(start_index, color_id_loc).set_lower_thresh(region.get_lower_thresh())
                global_region_scan.get_color_id_region(start_index, color_id_loc).set_upper_thresh(region.get_upper_thresh())
                global_region_scan.get_color_id_region(start_index, color_id_loc).track_and_seed()
                region_points = global_region_scan.get_color_id_region(start_index, color_id_loc).get_visited_pixel_list()
            else:
                print("No matching color found, checking another slice!")
            start_index -= 1

        start_index = global_region_scan.get_current_slice_index()
        start_index += 1
        print(global_region_scan)
        #while start_index < (global_region_scan.get_end_of_id(color_id_loc)):
        while start_index < (global_region_scan.get_z()):
            if global_region_scan.get_color_id_region(start_index, color_id_loc):
                global_region_scan.delete_color_region(start_index, color_id_loc)
            for pixel in region_points:
                if region.get_lower_thresh() <= global_region_scan.get_voxel_from_cube(pixel[0], pixel[1], start_index) <= region.get_upper_thresh():
                    slice_click(pixel[0], pixel[1], start_index, global_region_scan, color_id_loc)
            similar_colored_voxels = global_region_scan.get_threshed_slice(start_index, region.get_upper_thresh(), region.get_lower_thresh())
            min_dist = 4096
            min_index = -1
            for pixel_index in range(len(region_points)):
                for sim_index in range(len(similar_colored_voxels[0])):
                    distance = math.sqrt((region_points[pixel_index][0]-similar_colored_voxels[0][sim_index])**2+(region_points[pixel_index][1]-similar_colored_voxels[1][sim_index])**2)
                    if distance < min_dist:
                        min_dist = distance
                        min_index = sim_index
                    if min_dist == 0:
                        break
            if min_index != -1:
                slice_click(similar_colored_voxels[0][min_index], similar_colored_voxels[1][min_index], start_index, global_region_scan, color_id_loc)
                global_region_scan.get_color_id_region(start_index, color_id_loc).set_lower_thresh(region.get_lower_thresh())
                global_region_scan.get_color_id_region(start_index, color_id_loc).set_upper_thresh(region.get_upper_thresh())
                global_region_scan.get_color_id_region(start_index, color_id_loc).track_and_seed()
                region_points = global_region_scan.get_color_id_region(start_index, color_id_loc).get_visited_pixel_list()
            else:
                print("No matching color found, checking another slice!")
            start_index += 1
    update_canvas()


def near_trace_current(global_region_scan):
    print("Near-Trace current ID")
    global color_id
    region = global_region_scan.get_current_color_id_region(color_id)
    print("Working region:", region.get_color_id())
    start_index = global_region_scan.get_current_slice_index()
    region_points = region.get_visited_pixel_list()
    print("Found:", len(region_points), "visited Pixels")
    start_index -= 1
    lowest_dist = 4096
    while start_index >= 0:
        if global_region_scan.get_color_id_region(start_index, color_id):
            global_region_scan.delete_color_region(start_index, color_id)
        for pixel in region_points:
            if region.get_lower_thresh() <= global_region_scan.get_voxel_from_cube(pixel[0], pixel[1], start_index) <= region.get_upper_thresh():
                slice_click(pixel[0], pixel[1], start_index, global_region_scan, color_id)
        similar_colored_voxels = global_region_scan.get_threshed_slice(start_index, region.get_upper_thresh(), region.get_lower_thresh())
        min_dist = 4096
        min_index = -1
        for pixel_index in range(len(region_points)):
            for sim_index in range(len(similar_colored_voxels[0])):
                distance = math.sqrt((region_points[pixel_index][0] - similar_colored_voxels[0][sim_index]) ** 2 + (region_points[pixel_index][1] - similar_colored_voxels[1][sim_index]) ** 2)
                if distance < min_dist:
                    min_dist = distance
                    min_index = sim_index
                if min_dist == 0:
                    break
        if min_index != -1:
            slice_click(similar_colored_voxels[0][min_index], similar_colored_voxels[1][min_index], start_index, global_region_scan, color_id)
            global_region_scan.get_color_id_region(start_index, color_id).set_lower_thresh(region.get_lower_thresh())
            global_region_scan.get_color_id_region(start_index, color_id).set_upper_thresh(region.get_upper_thresh())
            global_region_scan.get_color_id_region(start_index, color_id).track_and_seed()
            region_points = global_region_scan.get_color_id_region(start_index, color_id).get_visited_pixel_list()
        else:
            print("No matching color found, checking another slice!")
        start_index -= 1
    start_index = global_region_scan.get_current_slice_index()
    start_index += 1
    while start_index < global_region_scan.get_z():
        if global_region_scan.get_color_id_region(start_index, color_id):
            global_region_scan.delete_color_region(start_index, color_id)
        for pixel in region_points:
            if region.get_lower_thresh() <= global_region_scan.get_voxel_from_cube(pixel[0], pixel[1], start_index) <= region.get_upper_thresh():
                slice_click(pixel[0], pixel[1], start_index, global_region_scan, color_id)
        similar_colored_voxels = global_region_scan.get_threshed_slice(start_index, region.get_upper_thresh(), region.get_lower_thresh())
        min_dist = 4096
        min_index = -1
        for pixel_index in range(len(region_points)):
            for sim_index in range(len(similar_colored_voxels[0])):
                distance = math.sqrt((region_points[pixel_index][0] - similar_colored_voxels[0][sim_index]) ** 2 + (region_points[pixel_index][1] - similar_colored_voxels[1][sim_index]) ** 2)
                if distance < min_dist:
                    min_dist = distance
                    min_index = sim_index
                if min_dist == 0:
                    break
        if min_index != -1:
            slice_click(similar_colored_voxels[0][min_index], similar_colored_voxels[1][min_index], start_index, global_region_scan, color_id)
            global_region_scan.get_color_id_region(start_index, color_id).set_lower_thresh(region.get_lower_thresh())
            global_region_scan.get_color_id_region(start_index, color_id).set_upper_thresh(region.get_upper_thresh())
            global_region_scan.get_color_id_region(start_index, color_id).track_and_seed()
            region_points = global_region_scan.get_color_id_region(start_index, color_id).get_visited_pixel_list()
        else:
            print("No matching color found, checking another slice!")
        start_index += 1
    update_canvas()

def slice_click(x, y, z, global_region_scan, color_id):
    voxel_value = global_region_scan.get_voxel_from_cube(x, y, z)
    #print("Slice: check at", x, y, z, voxel_value)
    if voxel_value > 0 and not global_region_scan.get_color_id_region(z, color_id):
        #print("NR:", color_id)
        current_region = Region(global_region_scan, color_id, z, global_region_scan.get_slice(z))
        current_region.add_seed(x, y)
        global_region_scan.append_region(z, current_region)
    elif voxel_value > 0 and global_region_scan.get_color_id_region(z, color_id):
        #print("SA:", color_id)
        global_region_scan.get_color_id_region(z, color_id).add_seed(x, y)


def current_click(x, y, global_region_scan):
    global color_id
    voxel_value = global_region_scan.get_voxel_from_slice(x, y)
    print("LM: clicked at", x, y, voxel_value)
    if voxel_value > 0 and not global_region_scan.get_current_color_id_region(color_id):
        print("NR")
        current_region = Region(global_region_scan, color_id, global_region_scan.get_current_slice_index(), global_region_scan.get_current_slice())
        current_region.add_seed_track(x, y)
        global_region_scan.current_append_region(current_region)
    elif voxel_value > 0 and global_region_scan.get_current_color_id_region(color_id):
        print("SA")
        global_region_scan.get_current_color_id_region(color_id).add_seed_track(x, y)
    update_canvas()


def left_mouse(event, global_region_scan):
    if 0 < event.x < global_region_scan.get_x() and 0 < event.y < global_region_scan.get_y():
        current_click(event.x, event.y, global_region_scan)
    elif global_region_scan.get_x() + 20 < event.x < global_region_scan.get_x() + 50 and 62 < event.y < 62 + (2 * 255):
        if global_region_scan.get_current_color_id_region(color_id):
            value = int((event.y - 62) / 2)
            print("TWC:", value)
            if global_region_scan.get_current_color_id_region(color_id).get_lower_thresh() < value < global_region_scan.get_current_color_id_region(color_id).get_upper_thresh():
                if value - global_region_scan.get_current_color_id_region(color_id).get_lower_thresh() < global_region_scan.get_current_color_id_region(color_id).get_upper_thresh() - value:
                    global_region_scan.get_current_color_id_region(color_id).set_lower_thresh(value)
                else:
                    global_region_scan.get_current_color_id_region(color_id).set_upper_thresh(value)
            elif value < global_region_scan.get_current_color_id_region(color_id).get_lower_thresh():
                global_region_scan.get_current_color_id_region(color_id).set_lower_thresh(value)
            elif value > global_region_scan.get_current_color_id_region(color_id).get_upper_thresh():
                global_region_scan.get_current_color_id_region(color_id).set_upper_thresh(value)
            update_canvas()


drag_array = []


def mouse_dragged(event, global_region_scan):
    global drag_array
    if 0 < event.x < global_region_scan.get_x() and 0 < event.y < global_region_scan.get_y():
        drag_array.append([event.x, event.y])


def left_mouse_up(global_region_scan):
    print("Release")
    global drag_array
    if len(drag_array) != 0:
        if global_region_scan.get_current_color_id_region(color_id):
            for point in drag_array:
                # print("Drag:", point[0], "---", point[1])
                voxel_value = global_region_scan.get_voxel_from_slice(point[0], point[1])
                if voxel_value > 0 and global_region_scan.get_current_color_id_region(color_id):
                    global_region_scan.get_current_color_id_region(color_id).add_seed(point[0], point[1])
            global_region_scan.get_current_color_id_region(color_id).track()
    drag_array = []
    update_canvas()


def right_mouse(event, global_region_scan):
    global color_id
    global_region_scan.delete_current_color_region(color_id)
    update_canvas()


def mouse_wheel(event, global_region_scan):
    global color_id
    # respond to Linux or Windows wheel event
    if global_region_scan.get_current_color_id_region(color_id):
        if (event.num == 4 or event.delta == -120) and event.state == 16:
            global_region_scan.get_current_color_id_region(color_id).change_upper_thresh(1)
        if (event.num == 4 or event.delta == 120) and event.state == 17:
            global_region_scan.get_current_color_id_region(color_id).change_lower_thresh(1)
        update_canvas()


def mouse_wheel_shift(event, global_region_scan):
    global color_id
    # respond to Linux or Windows wheel event
    if global_region_scan.get_current_color_id_region(color_id):
        if (event.num == 5 or event.delta == -120) and event.state == 16:
            global_region_scan.get_current_color_id_region(color_id).change_upper_thresh(-1)
        if (event.num == 5 or event.delta == 120) and event.state == 17:
            global_region_scan.get_current_color_id_region(color_id).change_lower_thresh(-1)
        update_canvas()


th_list = []

def save_mask(global_region_scan):
    file = filedialog.asksaveasfile(mode='w')
    print(file.name)
    mask_data = global_region_scan.get_global_region_mask_as_nifti_array()
    print(mask_data.shape)
    #scan_new_data = np.einsum('bac->abc', global_region_scan)
    imgScan = nb.Nifti1Image(mask_data, global_region_scan.get_scan_affine(), global_region_scan.get_scan_header())
    nb.save(imgScan, file.name)
    return

def save_dcm_mask(global_region_scan):
    file = filedialog.asksaveasfile(mode='w')
    print(file.name)
    dataset = global_region_scan.get_global_region_mask_as_dicom_array()
    dataset.save_as(file.name)

def save_dcm_scan(global_region_scan):
    file = filedialog.asksaveasfile(mode='w')
    print(file.name)
    dataset = global_region_scan.get_global_region_scan_as_dicom_array()
    dataset.save_as(file.name)

def save_scan(global_region_scan):
    file = filedialog.asksaveasfile(mode='w')
    print(file.name)
    mask_data = global_region_scan.get_global_region_scan_as_nifti_array()
    print(mask_data.shape)
    #scan_new_data = np.einsum('bac->abc', global_region_scan)
    imgScan = nb.Nifti1Image(mask_data, global_region_scan.get_scan_affine(), global_region_scan.get_scan_header())
    nb.save(imgScan, file.name)
    return
switch = False


def update_canvas():
    global switch
    global th_list
    global main_frame
    global global_region_scan
    canvas_repaint()
    for item in th_list:
        canvas.delete(item)
    th_list = []
    for region in global_region_scan.get_current_regions():
        print("Reg.", region.get_color_id())
        olc = translate_color_id(region.get_color_id())
        if not switch:
            canvas.create_image((0, 0), anchor="nw", image=region.get_image())
        th_list.append(canvas.create_rectangle(global_region_scan.get_x() + 10, 60 + int(2 * region.get_lower_thresh()), global_region_scan.get_x() + 30, 60 + int(2 * region.get_lower_thresh()), fill=olc, outline=""))
        th_list.append(canvas.create_text(global_region_scan.get_x() + 120, 60 + int(2 * region.get_lower_thresh()), fill=olc, text=str(int(region.get_lower_thresh())), font="Times 12"))
        th_list.append(canvas.create_rectangle(global_region_scan.get_x() + 10, 60 + int(2 * region.get_upper_thresh()), global_region_scan.get_x() + 30, 60 + int(2 * region.get_upper_thresh()), fill=olc, outline=""))
        th_list.append(canvas.create_text(global_region_scan.get_x() + 120, 60 + int(2 * region.get_upper_thresh()), fill=olc, text=str(int(region.get_upper_thresh())), font="Times 12"))
    canvas.update()


def toggle_switch(global_region_scan):
    global switch
    switch = not switch
    update_canvas()


def generate_stats(global_region_scan, median_image, blown_up_array):
    collapsed_col, collapsed_bin, collapsed_points = collapse(median_image, blown_up_array)
    collapsed_col = stats(blown_up_array, collapsed_col, collapsed_points)
    #collapsed_col = find_id(1, colored_array, collapsed_col, regio_lock)
    #collapsed_col = find_id(2, colored_array, collapsed_col, regio_lock)
    show_color(collapsed_col)


counter = 0
main_frame = tk.Toplevel()
canvas = tk.Canvas(main_frame)


def load_dcm_as_array(path):
    ds = dcmread(str(path))
    print(ds.pixel_array.shape)
    return ds.pixel_array

def load_nii_as_array(path):
    nii_arr = nb.load(path).get_fdata()
    nii_arr = np.einsum("abc->cba", nii_arr)
    return nii_arr

def load_dcm_as_dataset(path):
    ds = dcmread(str(path))
    print(ds)
    return ds

def is_nifti_file(filename):
    """Checks if a file is in NIfTI format by examining its magic numbers"""
    with open(filename, 'rb') as f:
        f.seek(344)
        magic = f.read(4)
        if magic == b'n+1\0':
            return True
        else:
            return False

def is_dicom_file(filename):
    """Checks if a file is in DICOM format by examining its magic numbers"""
    with open(filename, 'rb') as f:
        f.seek(128)
        magic = f.read(4)
        if magic.decode('utf-8') == 'DICM':
            return True
        else:
            return False

def stats_from_param():
    print("Stats...")
    scan_in_path = filedialog.askopenfilename(initialdir=init_dir, title="NORMALISIERTER SCAN DCM")
    mask_in_path = filedialog.askopenfilename(initialdir=init_dir, title="LABEL DCM/NII")
    dicom_in_path = filedialog.askopenfilename(initialdir="/", title="Original DCM/IMA")
    series_id = simpledialog.askstring(title="ID", prompt="Welche ID soll die Serie erhalten?")
    scan_array = load_dcm_as_array(scan_in_path)
    if os.path.isfile(mask_in_path):
        if is_nifti_file(mask_in_path):
            mask_array = load_nii_as_array(mask_in_path)
            print('The file is in NIfTI format')
        elif is_dicom_file(mask_in_path):
            mask_array = load_dcm_as_array(mask_in_path)
            print('The file is in DICOM format')
        else:
            print('The file is not in NIfTI or DICOM format')
            exit(-1)
    else:
        print('The file does not exist')
        exit(-1)

    dh = load_dcm_as_dataset(dicom_in_path)

    scan_array = np.einsum("abc->bca",scan_array)
    mask_array = np.einsum("abc->bca",mask_array)
    #print("SAS:",scan_array.shape)
    #print("MAS:",mask_array.shape)
    empty = np.zeros_like(scan_array)
    vessel = copy.deepcopy(scan_array)
    vessel[vessel > 0] = 255
    vessel[vessel <= 0] = 0
    show_mono(vessel[:,:,vessel.shape[2]//2])
    collapsed_col, collapsed_bin, collapsed_points  = collapse(vessel[:,:,vessel.shape[2]//2], vessel[:,:,vessel.shape[2]//2])
    collapsed_col = stats(vessel[:,:,vessel.shape[2]//2], collapsed_col, collapsed_points)
    vessel_coords = np.where(vessel==255)
    max_x = np.amax(vessel_coords[0])
    min_x = np.amin(vessel_coords[0])
    max_y = np.amax(vessel_coords[1])
    min_y = np.amin(vessel_coords[1])
    #print("miX:",min_x,", maX:",max_x,", miY:",min_y,", maY:",max_y)
    collapsed_col, path_one = find_id(1, mask_array, collapsed_col, translate_color_id_to_rgb(1-1), [[min_x, max_x],[min_y, max_y]], "biggest")
    collapsed_col, path_two = find_id(2, mask_array, collapsed_col, translate_color_id_to_rgb(2-1), [[min_x, max_x],[min_y, max_y]], "biggest")
    collapsed_col, path_three = find_id(3, mask_array, collapsed_col, translate_color_id_to_rgb(3-1), [[min_x, max_x],[min_y, max_y]], "biggest")
    print("P:",dicom_in_path)
    print("==============================================COPY FROM BELOW=========================================================")
    db_s = "INSERT INTO thromsom.meta (sID, orig_name, series_desc, study_date, study_time, acquisition_time, acquisition_number, distance_s2d, pxl_x, pxl_y, stenose_dia, stent_retr, stent_length, stent_dia, microcath, mc_dia, mc_length, fragmentation, stuck, bubbles, comment) values" \
           " ('" + str(series_id)  + "','UKN','" + str(dh[0x0008, 0x103e].value) + "','" + str(dh[0x0008, 0x0020].value) + "','" + str(dh[0x0008, 0x0030].value) + "','" + str(dh[0x0008, 0x0032].value) + \
           "','" + str(dh[0x0020, 0x0012].value) + "','" + str(dh[0x0018, 0x1110].value) + "','" + str(dh[0x0018, 0x1164][0]) + "','" + str(dh[0x0018, 0x1164][1]) + "',0,'" + str("") + "',0,0,'" + str("") + "',0,0,0,0,0,'" + str("") + "');"
    print(db_s)
    print("INSERT INTO thromsom.data (sID, slice_number, color_id, x,y,min_x, max_x, min_y, max_y, pixel_count, pixel_edges, pixel_sides) VALUES")
    analyze_id(series_id, 1, mask_array, path_one)
    analyze_id(series_id, 2, mask_array, path_two)
    analyze_id(series_id, 3, mask_array, path_three)
    print(";")
    #find_area(4, mask_array)
    #collapsed_col = find_id(4, mask_nifti, collapsed_col, translate_color_id_to_rgb(4-1), [[min_x, max_x],[min_y, max_y]], "biggest")
    show_color(collapsed_col)

def generate_stats():
    print("Stats...")
    scan_in_path = filedialog.askopenfilename(initialdir="/", title="Norm. Scan Dicom")
    mask_in_path = filedialog.askopenfilename(initialdir="/", title="Mask Dicom")
    dicom_in_path = filedialog.askopenfilename(initialdir="/", title="Original Dicom")
    series_id = simpledialog.askinteger(title="ID", prompt="Welche ID soll die Serie erhalten?")
    scan_array = load_dcm_as_array(scan_in_path)
    mask_array = load_dcm_as_array(mask_in_path)
    dh = load_dcm_as_dataset(dicom_in_path)

    scan_array = np.einsum("abc->bca",scan_array)
    mask_array = np.einsum("abc->bca",mask_array)
    #print("SAS:",scan_array.shape)
    #print("MAS:",mask_array.shape)
    empty = np.zeros_like(scan_array)
    vessel = copy.deepcopy(scan_array)
    vessel[vessel > 0] = 255
    vessel[vessel <= 0] = 0
    show_mono(vessel[:,:,vessel.shape[2]//2])
    collapsed_col, collapsed_bin, collapsed_points  = collapse(vessel[:,:,vessel.shape[2]//2], vessel[:,:,vessel.shape[2]//2])
    collapsed_col = stats(vessel[:,:,vessel.shape[2]//2], collapsed_col, collapsed_points)
    vessel_coords = np.where(vessel==255)
    max_x = np.amax(vessel_coords[0])
    min_x = np.amin(vessel_coords[0])
    max_y = np.amax(vessel_coords[1])
    min_y = np.amin(vessel_coords[1])
    #print("miX:",min_x,", maX:",max_x,", miY:",min_y,", maY:",max_y)
    collapsed_col, path_one = find_id(1, mask_array, collapsed_col, translate_color_id_to_rgb(1-1), [[min_x, max_x],[min_y, max_y]], "biggest")
    collapsed_col, path_two = find_id(2, mask_array, collapsed_col, translate_color_id_to_rgb(2-1), [[min_x, max_x],[min_y, max_y]], "biggest")
    collapsed_col, path_three = find_id(3, mask_array, collapsed_col, translate_color_id_to_rgb(3-1), [[min_x, max_x],[min_y, max_y]], "biggest")

    print("==============================================COPY FROM BELOW=========================================================")
    db_s = "INSERT INTO thromsom_db.meta (sID, series_desc, study_date, study_time, acquisition_time, acquisition_number, distance_s2d, pxl_x, pxl_y, stenose_dia, stent_retr, stent_length, stent_dia, microcath, mc_dia, mc_length, fragmentation, stuck, bubbles, comment) values" \
           " ('" + str(series_id) + "','" + str(dh[0x0008, 0x103e].value) + "','" + str(dh[0x0008, 0x0020].value) + "','" + str(dh[0x0008, 0x0030].value) + "','" + str(dh[0x0008, 0x0032].value) + \
           "','" + str(dh[0x0020, 0x0012].value) + "','" + str(dh[0x0018, 0x1110].value) + "','" + str(dh[0x0018, 0x1164][0]) + "','" + str(dh[0x0018, 0x1164][1]) + "',0,'" + str("") + "',0,0,'" + str("") + "',0,0,0,0,0,'" + str("") + "');"
    print(db_s)
    print("INSERT INTO thromsom_db.data (sID, slice_number, color_id, x,y,min_x, max_x, min_y, max_y, thrombus_count, edge_voxels, sides) VALUES")
    analyze_id(series_id, 1, mask_array, path_one)
    analyze_id(series_id, 2, mask_array, path_two)
    analyze_id(series_id, 3, mask_array, path_three)
    print(";")
    #find_area(4, mask_array)
    #collapsed_col = find_id(4, mask_nifti, collapsed_col, translate_color_id_to_rgb(4-1), [[min_x, max_x],[min_y, max_y]], "biggest")
    show_color(collapsed_col)

def track_gui(imgScan, median_array, blown_up_array):
    global canvas
    global main_frame
    global img
    global index_label
    global color_id
    global color_id_sv
    global global_region_scan
    main_frame = tk.Tk()
    global_region_scan = global_region(imgScan, main_frame)

    color_id_label = tk.StringVar()
    main_frame.title("ThromSomTrack")
    #main_frame.geometry(str(global_region_scan.get_x() + 200) + "x" + str(global_region_scan.get_y()))
    main_frame.geometry(str(global_region_scan.get_x() + 200) + "x" + str(1024))
    #main_frame.geometry("200x200")
    main_frame.resizable(0, 0)
    img = global_region_scan.get_current_region_image()
    print(img.width(), "--", img.height())
    canvas_width = (global_region_scan.get_x() + 200)
    #canvas_height = (global_region_scan.get_y())
    canvas_height = (1024)
    canvas = tk.Canvas(main_frame, width=canvas_width, height=canvas_height)
    color_id_sv = tk.StringVar(main_frame, str(0))
    color_id_sv.set(str(0))
    main_frame.bind("<Key>", lambda event: key_press(event, global_region_scan))
    canvas.bind("<Button-1>", lambda event: left_mouse(event, global_region_scan))
    canvas.bind("<ButtonRelease-1>", lambda event: left_mouse_up(global_region_scan))
    canvas.bind("<Button-3>", lambda event: right_mouse(event, global_region_scan))
    canvas.bind("<B1-Motion>", lambda event: mouse_dragged(event, global_region_scan))
    main_frame.bind("<Button-4>", lambda event: mouse_wheel(event, global_region_scan))
    main_frame.bind("<Button-5>", lambda event: mouse_wheel_shift(event, global_region_scan))
    load_button = tk.Button(canvas, text="Laden", command=stats, state=DISABLED)
    load_button.place(x=global_region_scan.get_x()+104, y=12, anchor=W, width=100, height=20)
    CreateToolTip(load_button, "Laden von fertigen Masken (noch nicht verfügbar)")
    save_mask_button = tk.Button(canvas, text="Maske sp.", command=lambda: save_dcm_mask(global_region_scan))
    save_mask_button.place(x=global_region_scan.get_x()+2, y=12, anchor=W, width=100, height=20)
    CreateToolTip(save_mask_button, "Speichert die Maske")
    save_scan_button = tk.Button(canvas, text="Scan sp.", command=lambda: save_dcm_scan(global_region_scan))
    save_scan_button.place(x=global_region_scan.get_x()+2, y=32, anchor=W, width=100, height=20)
    CreateToolTip(save_scan_button, "Speichert den nachbearbeiteten Scan")
    mode_button_dec = tk.Button(canvas, text="-", command=decrease_cid)
    mode_button_dec.place(x=global_region_scan.get_x(), y=54, anchor=W, width=50, height=20)
    CreateToolTip(mode_button_dec, "Farb-ID reduzieren")
    index_label = tk.Label(canvas, textvariable=color_id_sv, fg=translate_color_id(0), bg="#000")
    index_label.place(x=global_region_scan.get_x() + 60, y=54, anchor=W)
    mode_button_inc = tk.Button(canvas, text="+", command=increase_cid)
    mode_button_inc.place(x=global_region_scan.get_x() + 100, y=54, anchor=W, width=50, height=20)
    CreateToolTip(mode_button_inc, "Farb-ID erhöhen")

    mode_button_end = tk.Button(canvas, text="E", command=lambda: global_region_scan.set_end_of_id(color_id))
    mode_button_end.place(x=global_region_scan.get_x() + 150, y=54, anchor=W, width=50, height=20)
    CreateToolTip(mode_button_end, "Letz. Slice m. vollst. Obj.")


    #menu_button_1 = tk.Button(canvas, text="Z-One", command=increase_cid)
    #menu_button_1.place(x=global_region_scan.get_x() + 10, y=(512+62+10), anchor=W, width=50, height=50)

    #menu_button_2 = tk.Button(canvas, text="Z-All", command=lambda: z_trace(global_region_scan))
    #menu_button_2.place(x=global_region_scan.get_x() + 70, y=(512+62+10), anchor=W, width=100, height=20)

    menu_button_3 = tk.Button(canvas, text="Z-Trace", command=lambda: long_trace(global_region_scan), state=DISABLED)
    menu_button_3.place(x=global_region_scan.get_x() + 2, y=(512+62+70), anchor=W, width=100, height=20)
    CreateToolTip(menu_button_3, "Tracing auf Z-Achse: Findet alle farblich gleichen Regionen anderer Schichten, die die Regionen der aktuellen Schicht berühren")

    menu_button_4 = tk.Button(canvas, text="Slice-Trace", command=lambda: slice_trace(global_region_scan), state=DISABLED)
    menu_button_4.place(x=global_region_scan.get_x() + 2, y=(512 + 62 + 92), anchor=W, width=100, height=20)
    CreateToolTip(menu_button_4, "Tracing auf Schicht: Findet alle farblich gleichen Regionen auf der aktuellen Schicht")

    menu_button_5 = tk.Button(canvas, text="All-Slice-Trace", command=lambda: all_slice_trace(global_region_scan), state=DISABLED)
    menu_button_5.place(x=global_region_scan.get_x() + 104, y=(512 + 62 + 92), anchor=W, width=100, height=20)
    CreateToolTip(menu_button_5, "Tracing auf allen Schichten: Findet alle farblich gleichen Regionen auf allen Schichten")

    menu_button_6 = tk.Button(canvas, text="Near-Trace", command=lambda: near_trace(global_region_scan))
    menu_button_6.place(x=global_region_scan.get_x() + 2, y=(512 + 62 + 114), anchor=W, width=100, height=20)
    CreateToolTip(menu_button_6, "Tracing auf allen Schichten: Findet alle farblich gleichen Regionen auf allen Schichten, wählt die nahestgelegene, folgende Region als nächsten Seedpoint aus")

    menu_button_8 = tk.Button(canvas, text="Maske an/aus", command=lambda: toggle_switch(global_region_scan))
    menu_button_8.place(x=global_region_scan.get_x() + 104, y=(512 + 62 + 114), anchor=W, width=100, height=20)
    CreateToolTip(menu_button_8, "Stellt die Maske an/aus")

    menu_button_9 = tk.Button(canvas, text="Near-Trace-Cur", command=lambda: near_trace_current(global_region_scan), state=DISABLED)
    menu_button_9.place(x=global_region_scan.get_x() + 2, y=(512 + 62 + 136), anchor=W, width=100, height=20)
    CreateToolTip(menu_button_9, "Tracing auf allen Schichten: Wie Near-Trace, nur mit der oben gewählten Farbe")


    #menu_button_4 = tk.Button(canvas, text="Region-Seed", command=increase_cid)
    #menu_button_4.place(x=global_region_scan.get_x() + 70, y=(512+62+70), anchor=W, width=100, height=20)

    #menu_button_5 = tk.Button(canvas, text="Save", command=lambda: save_mask(global_region_scan))
    #menu_button_5.place(x=global_region_scan.get_x() + 10, y=(512+62+130), anchor=W, width=50, height=50)

    menu_button_7 = tk.Button(canvas, text="Stats", command=lambda: generate_stats(global_region_scan, median_array, blown_up_array))
    menu_button_7.place(x=global_region_scan.get_x() + 104, y=(512+62+70), anchor=W, width=100, height=20)
    CreateToolTip(menu_button_7, "Erstellt Statistiken")

    for i in range(256):
        hex_string = hex(i)[2:]
        if i < 16:
            col = '#' + "0" + hex_string + "0" + hex_string + "0" + hex_string
        else:
            col = '#' + hex_string + hex_string + hex_string
        canvas.create_rectangle(global_region_scan.get_x() + 20, 60 + (2 * i), global_region_scan.get_x() + 50, 62 + (2 * i), fill=col, outline="")

    canvas.focus_get()
    canvas.pack()
    canvas.create_image((0, 0), anchor="nw", image=global_region_scan.get_current_region_image())
    canvas.update()

    # canvas_repaint(imgScan)

    main_frame.mainloop()
    return None


def canvas_repaint():
    global canvas
    global counter
    global main_frame
    global img
    global global_region_scan
    canvas.pack()
    canvas.create_image((0, 0), anchor="nw", image=global_region_scan.get_current_region_image())
    canvas.update()



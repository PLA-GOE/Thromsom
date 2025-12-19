from tkinter import Image

import numpy as np
import copy

from PIL import Image, ImageTk


current_id = 0


class Region:

    def __init__(self, global_region_scan, color_id, z, current_slice):
        global current_id
        self.master = global_region_scan.get_master()
        self.append_id = -1
        self.id = current_id
        current_id += 1
        self.color_id = color_id
        self.z = z
        self.current_slice = current_slice
        self.seed_list = []
        self.upper_thresh = -1
        self.lower_thresh = 256
        self.visited_array = np.zeros_like(self.current_slice).astype(np.uint8)
        print(self.z,": shape:",self.visited_array.shape[0],"---",self.visited_array.shape[1],"---",4)
        self.image_array = np.zeros((self.visited_array.shape[0], self.visited_array.shape[1], 4),)
        print(self.image_array.shape)
        self.visited_pixel_list = []
        self.image = ImageTk.PhotoImage(master=self.master, image=Image.fromarray(self.image_array, mode="RGBA"))

    def track_region(self):
        self.seed_list = []
        same_color_array = np.where((self.current_slice>=self.lower_thresh) & (self.current_slice<=self.upper_thresh))
        #print(same_color_array)
        for point_index in range(len(same_color_array[0])):
            self.add_seed(same_color_array[0][point_index], same_color_array[1][point_index])
        self.track()

    def track_region_blocked(self, blocked_array):
        self.seed_list = []
        same_color_array = np.where((self.current_slice>=self.lower_thresh) & (self.current_slice<=self.upper_thresh) & (blocked_array != 1))
        for point_index in range(len(same_color_array[0])):
            self.add_seed(same_color_array[0][point_index], same_color_array[1][point_index])
        self.track_and_seed_blocked(blocked_array)

    def track_region_blocked_th(self, blocked_array, th_lo, th_up):
        self.seed_list = []
        same_color_array = np.where((self.current_slice>=th_lo) & (self.current_slice<=th_up) & (blocked_array != 1))
        for point_index in range(len(same_color_array[0])):
            self.add_seed(same_color_array[0][point_index], same_color_array[1][point_index])
        self.track_and_seed()

    def __str__(self):
        return "Region " + str(current_id) + ": CID:" + str(self.color_id) + ", z:" + str(self.z)

    def get_id(self):
        return self.id

    def get_color_id(self):
        return self.color_id

    def get_z(self):
        return self.z

    def get_voxels_in_thresh(self, upper_thresh, lower_thresh):
        return np.where((self.current_slice>=lower_thresh) & (self.current_slice<=upper_thresh))

    def get_seed_count(self):
        return len(self.seed_list)

    def add_seed(self, x, y):
        if [x, y] not in self.seed_list:
            self.seed_list.append([x, y])
            value = self.get_value(x, y)
            if value > 0:
                if value > self.upper_thresh:
                    self.upper_thresh = value
                if value < self.lower_thresh:
                    self.lower_thresh = value

    def add_seed_track(self, x, y):
        print("seeds:", len(self.seed_list))
        if [x, y] not in self.seed_list:
            self.seed_list.append([x, y])
            value = self.get_value(x, y)
            if value > 0:
                if value > self.upper_thresh:
                    self.upper_thresh = value
                if value < self.lower_thresh:
                    self.lower_thresh = value
                self.track()

    def get_value(self, x, y):
        return self.current_slice[x, y]

    def change_lower_thresh(self, value):
        if self.lower_thresh + value >= 0:
            self.lower_thresh += value
        self.track()

    def get_lower_thresh(self):
        return self.lower_thresh

    def set_lower_thresh(self, value):
        self.lower_thresh = int(value)
        self.track()

    def change_upper_thresh(self, value):
        if self.upper_thresh + value <= 255:
            self.upper_thresh += value
        self.track()

    def get_upper_thresh(self):
        return self.upper_thresh

    def set_upper_thresh(self, value):
        self.upper_thresh = int(value)
        self.track()

    def get_image(self):
        #print("Image call")
        #print(self.image.width(),"/",self.image.height())
        #print(self.image)
        return self.image

    def track(self):
        color = translate_color_id_to_rgba(self.color_id)
        th_list = copy.deepcopy(self.seed_list)
        pixel_list = []
        for pixel_tuple_index in range(len(th_list)):
            value = self.get_value(th_list[pixel_tuple_index][0], th_list[pixel_tuple_index][1])
            if self.lower_thresh <= value <= self.upper_thresh:
                pixel_list.append((th_list[pixel_tuple_index][0], th_list[pixel_tuple_index][1]))
        self.visited_array = np.zeros_like(self.current_slice)
        self.visited_pixel_list = []
        self.image_array = np.zeros((self.visited_array.shape[0], self.visited_array.shape[1], 4),)
        while pixel_list:
            pixel = pixel_list.pop()
            if self.visited_array[pixel[0]][pixel[1]] == 0:
                self.visited_array[pixel[0]][pixel[1]] = 1
                self.visited_pixel_list.append([pixel[0], pixel[1]])
                pixel_list.extend(self.get_neighbours(pixel[0], pixel[1]))
        self.image_array[self.visited_array == 1] = np.asarray(color).astype(np.uint8)
        self.image = ImageTk.PhotoImage(master=self.master, image=Image.fromarray(np.fliplr(np.rot90(self.image_array.astype(np.uint8), axes=(1, 0))), "RGBA"))

    def track_and_seed(self):
        color = translate_color_id_to_rgba(self.color_id)
        pixel_list = copy.deepcopy(self.seed_list)
        self.visited_array = np.zeros_like(self.current_slice)
        self.visited_pixel_list = []
        self.image_array = np.zeros((self.visited_array.shape[0], self.visited_array.shape[1], 4), )
        while pixel_list:
            pixel = pixel_list.pop()
            if self.visited_array[pixel[0]][pixel[1]] == 0:
                self.visited_array[pixel[0]][pixel[1]] = 1
                self.visited_pixel_list.append([pixel[0], pixel[1]])
                self.seed_list.append([pixel[0], pixel[1]])
                pixel_list.extend(self.get_neighbours(pixel[0], pixel[1]))
        self.image_array[self.visited_array == 1] = np.asarray(color).astype(np.uint8)
        self.image = ImageTk.PhotoImage(master=self.master, image=Image.fromarray(np.fliplr(np.rot90(self.image_array.astype(np.uint8), axes=(1, 0))), "RGBA"))

    def track_and_seed_blocked(self, blocked_array):
        color = translate_color_id_to_rgba(self.color_id)
        pixel_list = copy.deepcopy(self.seed_list)
        self.visited_array = (blocked_array*-1)
        self.visited_pixel_list = []
        self.image_array = np.zeros((self.visited_array.shape[0], self.visited_array.shape[1], 4), )
        while pixel_list:
            pixel = pixel_list.pop()
            if self.visited_array[pixel[0]][pixel[1]] == 0:
                self.visited_array[pixel[0]][pixel[1]] = 1
                self.visited_pixel_list.append([pixel[0], pixel[1]])
                self.seed_list.append([pixel[0], pixel[1]])
                pixel_list.extend(self.get_neighbours(pixel[0], pixel[1]))
        self.image_array[self.visited_array == 1] = np.asarray(color).astype(np.uint8)
        self.image = ImageTk.PhotoImage(master=self.master, image=Image.fromarray(np.fliplr(np.rot90(self.image_array.astype(np.uint8), axes=(1, 0))), "RGBA"))

    def slice_recon(self):
        pixel_list = copy.deepcopy(self.seed_list)
        self.visited_array = np.zeros_like(self.current_slice)
        self.visited_pixel_list = []
        while pixel_list:
            pixel = pixel_list.pop()
            if self.visited_array[pixel[0]][pixel[1]] == 0:
                self.visited_array[pixel[0], pixel[1]] = 1
                self.visited_pixel_list.append([pixel[0], pixel[1]])
                pixel_list.extend(self.get_neighbours(pixel[0], pixel[1]))

    def where(self, upper_thresh, lower_thresh):
        return np.where(self.current_slice>=lower_thresh & self.current_slice<=upper_thresh)

    def get_visited_array(self):
        return self.visited_array

    #def delete_pixels(self):
    #    self.seed_list = []
    #    self.visited_array = np.zeros_like(self.current_slice).astype(np.uint8)
    #    self.visited_pixel_list = []
    #    self.image_array = np.zeros((self.visited_array.shape[0], self.visited_array.shape[1], 4), )
    #    self.image = ImageTk.PhotoImage(image=Image.fromarray(np.fliplr(np.rot90(self.image_array.astype(np.uint8), axes=(1, 0))), "RGBA"))

    def set_append_id(self, append_id):
        self.append_id = append_id

    def get_append_id(self):
        return self.append_id

    def get_visited_color_array(self):
        print(self.visited_array*self.color_id)
        return (self.visited_array*(self.color_id+1)).astype('uint16', casting='unsafe')

    def get_visited_pixel_list(self):
        return self.visited_pixel_list

    def get_neighbours(self, x, y):
        return_array = []
        if x < self.current_slice.shape[0]-1 and y < self.current_slice.shape[1]-1 and x > 0 and y > 0:
            vxp1yp1 = self.current_slice[x + 1][y + 1]
            vxp1y = self.current_slice[x + 1][y]
            vxyp1ym1 = self.current_slice[x + 1][y - 1]
            vxyp1 = self.current_slice[x][y + 1]
            vxym1 = self.current_slice[x][y - 1]
            vxm1yp1 = self.current_slice[x - 1][y + 1]
            vxm1y = self.current_slice[x - 1][y]
            vxm1ym1 = self.current_slice[x - 1][y - 1]
            if self.lower_thresh <= vxp1yp1 <= self.upper_thresh:
                return_array.append([x + 1, y + 1])
            if self.lower_thresh <= vxp1y <= self.upper_thresh:
                return_array.append([x + 1, y])
            if self.lower_thresh <= vxyp1ym1 <= self.upper_thresh:
                return_array.append([x + 1, y - 1])
            if self.lower_thresh <= vxyp1 <= self.upper_thresh:
                return_array.append([x, y + 1])
            if self.lower_thresh <= vxym1 <= self.upper_thresh:
                return_array.append([x, y - 1])
            if self.lower_thresh <= vxm1yp1 <= self.upper_thresh:
                return_array.append([x - 1, y + 1])
            if self.lower_thresh <= vxm1y <= self.upper_thresh:
                return_array.append([x - 1, y])
            if self.lower_thresh <= vxm1ym1 <= self.upper_thresh:
                return_array.append([x - 1, y - 1])
        return return_array

def translate_color_id(color_id):
    color_dict = {0: "#80ae80", 1: "#f1d691", 2: "#b17a65", 3: "#6fb8d2", 4: "#d8654f", 5: "#dd8265", 6: "#90ee90", 7: "#c06858", 8: "#dcf514", 9: "#44563b"}
    return color_dict.get(color_id)

def translate_color_id_to_rgb(color_id):
    color_dict = {0: [128, 174, 128],
                  1: [241, 214, 145],
                  2: [177, 122, 101],
                  3: [111, 184, 210],
                  4: [216, 101, 79],
                  5: [221, 130, 101],
                  6: [144, 238, 144],
                  7: [192, 104, 88],
                  8: [220, 245, 20],
                  9: [68, 86, 59]}
    return color_dict.get(color_id)

def translate_color_id_to_rgba(color_id):
    color_dict = {0: [128, 174, 128,255],
                  1: [241, 214, 145, 255],
                  2: [177, 122, 101,255],
                  3: [111, 184, 210,255],
                  4: [216, 101, 79, 255],
                  5: [221, 130, 101,255],
                  6: [144, 238, 144,255],
                  7: [192, 104, 88, 255],
                  8: [220, 245, 20,255],
                  9: [68, 86, 59,255]}
    return color_dict.get(color_id)
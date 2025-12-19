import copy

import numpy
import numpy as np

current_slice = 0

from PIL import Image, ImageTk


class global_region():

    def __init__(self, scan_cube, main_frame):
        from obj.gui import dataset
        self.dataset = dataset
        from obj.gui import global_cut_params
        self.global_cut_params = global_cut_params
        print("DS:", dataset)
        self.main_frame = main_frame
        self.scan_cube = scan_cube
        self.x = scan_cube.shape[0]
        self.y = scan_cube.shape[1]
        self.z = scan_cube.shape[2]
        self.slice_region_collector = [[] for _ in range(self.z)]
        self.max = np.amax(scan_cube)
        self.min = np.amax(scan_cube)
        self.current_image = None
        self.end_of_id = [scan_cube.shape[2], scan_cube.shape[2], scan_cube.shape[2], scan_cube.shape[2], scan_cube.shape[2], scan_cube.shape[2], scan_cube.shape[2], scan_cube.shape[2], scan_cube.shape[2], scan_cube.shape[2]]
        print(self.end_of_id)

    def __str__(self):
        return "GR: gcp:"+str(self.global_cut_params)+", x:"+str(self.x)+", y:"+str(self.y)+", z:"+str(self.z)+", max:"+str(self.max)+", min:"+str(self.min)+", eoi:"+str(self.end_of_id)

    def set_end_of_id(self, cid):
        global current_slice
        self.end_of_id[cid] = current_slice
        print(self.end_of_id[cid])

    def get_end_of_id(self, cid):
        print(self.end_of_id[cid])
        return self.end_of_id[cid]

    def get_master(self):
        return self.main_frame

    def get_current_slice_index(self):
        global current_slice
        return current_slice

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

    def get_max(self):
        return self.max

    def get_min(self):
        return self.min

    def get_shape(self):
        return self.scan_cube.shape

    def get_current_slice(self):
        global current_slice
        return self.scan_cube[:, :, current_slice]

    def get_current_region_image(self):
        self.current_image = ImageTk.PhotoImage(master=self.main_frame, image=Image.fromarray(np.fliplr(np.rot90(self.get_current_slice(), axes=(1, 0)))))
        return self.current_image

    def get_slice(self, index):
        return self.scan_cube[:, :, index]

    def get_threshed_slice(self, index, upper_thresh, lower_thresh):
        return np.where((self.scan_cube[:, :, index] >= lower_thresh) & (self.scan_cube[:, :, index] <= upper_thresh))

    def get_current_region_count(self):
        global current_slice
        return len(self.slice_region_collector[current_slice])

    def get_region_count(self, slice_index):
        return len(self.slice_region_collector[slice_index])

    def get_voxel_from_slice(self, x, y):
        return self.scan_cube[x, y, current_slice]

    def get_voxel_from_cube(self, x, y, z):
        return self.scan_cube[x, y, z]

    def get_current_regions(self):
        global current_slice
        return self.slice_region_collector[current_slice]

    def get_all_visited_pixels(self):
        return_array = np.zeros_like(self.scan_cube[:, :, current_slice])
        for region in self.get_current_regions():
            return_array = np.logical_or(np.asarray(return_array), np.asarray(region.get_visited_array()))
        return return_array

    def get_all_visited_pixels_z(self, index):
        return_array = np.zeros_like(self.scan_cube[:, :, index])
        for region in self.get_regions(index):
            return_array = np.logical_or(np.asarray(return_array), np.asarray(region.get_visited_array()))
        return return_array

    def get_regions(self, region_index):
        return self.slice_region_collector[region_index]

    def get_current_color_id_region(self, color_id):
        global current_slice
        for region in self.slice_region_collector[current_slice]:
            if region.get_color_id() == color_id:
                return region
        return None

    def delete_current_color_region(self, color_id):
        global current_slice
        for region in self.slice_region_collector[current_slice]:
            if region.get_color_id() == color_id:
                print("Delete:", region.get_append_id())
                # region.delete_pixels()
                self.slice_region_collector[current_slice].pop(region.get_append_id())

    def delete_color_region(self, region_index, color_id):
        global current_slice
        for region in self.slice_region_collector[region_index]:
            if region.get_color_id() == color_id:
                print("Delete:", region.get_append_id())
                # region.delete_pixels()
                self.slice_region_collector[region_index].pop(region.get_append_id())

    def get_color_id_region(self, region_index, color_id):
        for region in self.slice_region_collector[region_index]:
            if region.get_color_id() == color_id:
                return region
        return None

    def set_slice(self, fix_value):
        global current_slice
        if 0 <= fix_value < self.z:
            current_slice = fix_value

    def increase_slice(self, inc_value):
        global current_slice
        if 0 <= current_slice + inc_value < self.z:
            current_slice += inc_value

    def append_region(self, index, add_region):
        append_id = len(self.slice_region_collector[index])
        self.slice_region_collector[index].append(add_region)
        add_region.set_append_id(append_id)

    def current_append_region(self, add_region):
        global current_slice
        append_id = len(self.slice_region_collector[current_slice])
        print("Append:", current_slice, "---", add_region)
        self.slice_region_collector[current_slice].append(add_region)
        add_region.set_append_id(append_id)

    def get_global_region_mask_as_nifti_array(self):
        return_cube = np.zeros_like(self.scan_cube)
        # print(return_cube.shape)
        for region_slice_index in range(len(self.slice_region_collector)):
            for region in self.slice_region_collector[region_slice_index]:
                # print(region.get_visited_array().shape,":",region.get_color_id())
                return_cube[region_slice_index, :, :] += region.get_visited_color_array()
        return_cube = np.asarray(return_cube).astype(np.uint8)
        # print(np.bincount(return_cube.flatten()))
        return return_cube

    def get_global_region_mask_as_dicom_array(self):
        copy_set = copy.deepcopy(self.dataset)
        return_cube = np.zeros_like(self.dataset.pixel_array)
        print(return_cube.shape)
        print(return_cube.dtype)
        print(self.dataset.pixel_array.dtype)
        print(self.dataset.pixel_array.shape)
        from obj.image_panel import global_cut_params_tra
        for region_slice_index in range(len(self.slice_region_collector)):
            for region in self.slice_region_collector[region_slice_index]:
                print(region.get_visited_array().shape, ":", region.get_color_id())
                print(return_cube[:, int(global_cut_params_tra[0][0]):int(global_cut_params_tra[1][0]), int(global_cut_params_tra[0][1]):int(global_cut_params_tra[1][1])].shape)
                print(int(global_cut_params_tra[0][1]), int(global_cut_params_tra[1][1]), int(global_cut_params_tra[0][0]), int(global_cut_params_tra[1][0]))
                return_cube[region_slice_index, int(global_cut_params_tra[0][1]):int(global_cut_params_tra[1][1]), int(global_cut_params_tra[0][0]):int(global_cut_params_tra[1][0])] += np.flipud(np.rot90(region.get_visited_color_array(), axes=(0, 1)))
        return_cube = np.asarray(return_cube).astype(np.uint16)
        copy_set.PixelData = return_cube.astype(np.uint16).tobytes()
        return copy_set

    def get_global_region_scan_as_dicom_array(self):
        copy_set = copy.deepcopy(self.dataset)
        return_cube = np.zeros_like(self.dataset.pixel_array)
        print(return_cube.shape)
        print(return_cube.dtype)
        print(self.dataset.pixel_array.dtype)
        print(self.dataset.pixel_array.shape)
        from obj.image_panel import global_cut_params_tra
        for slice_index in range(self.z):
            return_cube[slice_index, int(global_cut_params_tra[0][1]):int(global_cut_params_tra[1][1]), int(global_cut_params_tra[0][0]):int(global_cut_params_tra[1][0])] = np.flipud(np.rot90(self.get_slice(slice_index), axes=(0, 1)))
        return_cube = np.asarray(return_cube).astype(np.uint16)
        copy_set.PixelData = return_cube.astype(np.uint16).tobytes()
        # print(np.bincount(return_cube.flatten()))
        return copy_set

    def get_global_region_scan_as_nifti_array(self):
        return_cube = np.asarray((((self.scan_cube - np.amin(self.scan_cube)) / (np.amax(self.scan_cube) - np.amin(self.scan_cube))) * 255.0)).astype(np.uint8)
        return return_cube

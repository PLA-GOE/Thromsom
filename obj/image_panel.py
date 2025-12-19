import itk
import numpy as np
from PIL import Image, ImageTk

import level_set
import tracker


global_cut_params_tra = None

def show_mono(image_in):
    image_min = np.amin(image_in)
    show_image = (((image_in - image_min) / (np.amax(image_in) - image_min)) * 255.0).astype(np.uint8)
    img = Image.fromarray(show_image, 'L')
    img.show()

def fill_holes(image_in, its):
    for it in range(its):
        fill = []
        for x in range(1, len(image_in) - 1):
            for y in range(1, len(image_in[x]) - 1):
                if get_neighbours(image_in, x, y) >= 5:
                    fill.append((x, y))
        for fill_pix in fill:
            image_in[fill_pix[0], fill_pix[1]] = 1
    return image_in

def get_neighbours(image_in, x, y):
    return np.sum(image_in[x - 1:x + 1, y - 1:y + 1])

def blow_up(image_in, add_width, y_start, y_end):
    image_in = np.einsum("ab->ba", image_in)
    stepping = 5
    width = image_in.shape[0]
    print(image_in.shape)
    for y_coord in range(y_start, y_end):
        x_coord = int(width / 2)
        it_count = 1
        x_pos = -1
        x_neg = -1
        while x_coord >= 0 and x_coord < width:
            print("Check:",x_coord,",",y_coord)
            if image_in[x_coord, y_coord] == 1:
                print("Hit: x:", x_coord, ", y:", y_coord)
                x_pos = 0
                while (x_coord + x_pos) < width and image_in[x_coord + x_pos, y_coord] == 1:
                    x_pos += 1
                print("End:", (x_coord + x_pos))
                x_neg = 0
                while (x_coord - x_neg) > 0 and image_in[x_coord - x_neg, y_coord] == 1:
                    x_neg += 1
                print("End:", (x_coord - x_neg))
                break
            else:
                if it_count % 2 == 1:
                    x_coord += (stepping * it_count)
                    it_count += 1
                else:
                    x_coord += -(stepping * it_count)
                    it_count += 1
        if x_pos != -1 and x_neg != -1:
            x_end = x_coord + x_pos
            x_start = x_coord - x_neg
            print("Blowup from:", x_start, "to", x_end, "with", add_width)
            image_in[x_start - add_width:x_end + add_width, y_coord] = 1
    image_in = np.einsum("ba->ab", image_in)
    return image_in

def get_bins(image_in):
    print(np.unique(image_in.flatten()))
    print(np.bincount(image_in.flatten()))

def generate_images(line_vert, line_hor, center_point, image_shape, image_array):
    print(line_vert, line_hor, center_point, image_shape)
    print("Translating...")
    fact_x = image_shape[1]/1024
    fact_y = image_shape[2]/1024
    line_vert_tra_0 = int(line_vert[0]*fact_x)
    line_vert_tra_1 = int(line_vert[1]*fact_x)
    line_hor_tra_0 = int(line_hor[0]*fact_y)
    line_hor_tra_1 = int(line_hor[1]*fact_y)
    center_point_tra_x = int(center_point[0]*fact_x)
    center_point_tra_y = int(center_point[1]*fact_y)
    print(line_vert_tra_0, line_vert_tra_1, line_hor_tra_0, line_hor_tra_1, center_point_tra_x, center_point_tra_y)
    left_upper_corner = [-1,-1]
    right_lower_corner = [-1,-1]

    if line_vert_tra_0 > line_vert_tra_1:
        left_upper_corner[1] = line_vert_tra_1
        right_lower_corner[1] = line_vert_tra_0
    else:
        left_upper_corner[1] = line_vert_tra_0
        right_lower_corner[1] = line_vert_tra_1

    if line_hor_tra_0 > line_hor_tra_1:
        left_upper_corner[0] = line_hor_tra_1
        right_lower_corner[0] = line_hor_tra_0
    else:
        left_upper_corner[0] = line_hor_tra_0
        right_lower_corner[0] = line_hor_tra_1
    print("luc:",left_upper_corner)
    print("rlc:",right_lower_corner)
    print("ctra:",center_point_tra_x, center_point_tra_y)
    global global_cut_params_tra
    global_cut_params_tra = [left_upper_corner, right_lower_corner]
    cut_array = image_array[:,left_upper_corner[1]:right_lower_corner[1],left_upper_corner[0]:right_lower_corner[0]]

    print(cut_array.shape)
    #median_image = create_median(cut_array)
    median_image = create_perc(cut_array)
    #median_image = create_simple_median_value(cut_array)
    median_filtered_image = median_filter(median_image, 3)
    print(median_filtered_image.shape)
    min_arr = np.amax(median_filtered_image)
    median_filtered_image[:,0:4] = min_arr
    median_filtered_image[:,-5:] = min_arr
    median_filtered_image[0:4,:] = min_arr
    median_filtered_image[-5:,:] = min_arr
    print("min:",np.amin(median_filtered_image),", max:",np.amax(median_filtered_image))
    seed_x = center_point_tra_x-left_upper_corner[0]
    seed_y = center_point_tra_y-left_upper_corner[1]
    print("Seed_points:",seed_x, seed_y)

    show_mono(median_filtered_image)
    edge_detected_image = edge_detection(median_filtered_image, seed_y, seed_x)
    show_mono(edge_detected_image)
    filled_scan = fill_holes(edge_detected_image,3)
    get_bins(edge_detected_image)
    blown_up_array = blow_up(filled_scan, 4, 0, (filled_scan.shape[0] - 1))

    median_image[blown_up_array == 0] = 0
    median_min = np.amin(median_image[blown_up_array != 0])
    median_max = np.amax(median_image[blown_up_array != 0])

    print("Med_max:", median_max, ", Med_min:", median_min)
    three_d_bum = np.zeros_like(cut_array, dtype=np.uint32)
    for slice in range(three_d_bum.shape[0]):
        three_d_bum[slice, :, :] = blown_up_array

    scan_block = np.where(three_d_bum != 0, cut_array, 0)
    # show_mono(scan_block[:, :, 30])

    for slice in range(scan_block.shape[0]):
        scan_block[slice, :, :] = (scan_block[slice, :, :] + 4096) - median_image

    cut_array[three_d_bum == 0] = 0

    # show_mono(scan_block[:, :, 30])
    # show_mono(scan_block[:, :, 31])
    # show_mono(scan_block[:, :, 32])

    scan_block_min = np.amin(scan_block[three_d_bum != 0])
    scan_block_max = np.amax(scan_block[three_d_bum != 0])
    print(scan_block_min, "-", scan_block_max)

    scan_block = (((scan_block - scan_block_min) / (scan_block_max - scan_block_min)) * 4096).astype(int)
    scan_block = np.where(three_d_bum != 0, scan_block, 0)
    scan_block_min = np.amin(scan_block)
    scan_block_max = np.amax(scan_block)
    print(scan_block_min, "-", scan_block_max)

    scan_new_data = trim_to(scan_block, 0, 255)
    print(scan_new_data.shape)
    scan_new_data = np.einsum("abc->cba", scan_new_data)
    where_info_array = np.where(blown_up_array == 1)
    min_x_ind = np.amin(where_info_array[0]) - 1
    max_x_ind = np.amax(where_info_array[0]) + 1
    min_y_ind = np.amin(where_info_array[1]) - 1
    max_y_ind = np.amax(where_info_array[1]) + 1
    stent_coords, thromb_coords = tracker.track_gui(scan_new_data, median_image, blown_up_array)

    return

def create_median(scan_in):
    print("Creating longitudinal median of all scans.")
    median_image = np.median(scan_in[:, :, :], axis=(0))
    print("Median created.")
    return median_image

def create_perc(scan_in):
    print("Creating longitudinal median of all scans.")
    median_image_low = np.percentile(scan_in[:, :, :], 30, axis=(0))
    median_image_high = np.percentile(scan_in[:, :, :], 70, axis=(0))
    median_image = median_image_low if median_image_high > median_image_low else median_image_high
    print("Median created.")
    return median_image


def create_simple_median_value(scan_in):
    print("Creating longitudinal median of all scans.")
    median_image = np.median(scan_in[:, :, :], axis=(0))
    print(median_image)
    return_array = np.zeros_like(median_image)
    return_array[median_image != 0] = np.median(median_image)
    print(return_array)
    print("Median created.")
    return return_array


def trim_to(input_image_array, lo, hi):
    output_image_array = (input_image_array - np.amin(input_image_array)) / float((np.amax(input_image_array) - np.amin(input_image_array))) * (hi - lo) + lo
    return output_image_array


def median_filter(image_in, filter_size):
    print("Filtering: Median")
    median_min = np.amin(image_in)
    median_filtered = (((image_in - median_min) / (np.amax(image_in) - median_min)) * 255.0).astype(np.float32)
    PixelType = itk.F
    Dimension = 2
    ImageType = itk.Image[PixelType, Dimension]
    image = itk.GetImageFromArray(median_filtered, ttype=(ImageType,))
    medianFilter = itk.MedianImageFilter[ImageType, ImageType].New()
    medianFilter.SetInput(image)
    medianFilter.SetRadius(filter_size)
    filtered_image = itk.GetArrayViewFromImage(medianFilter.GetOutput())
    print("Median filtered.")
    return filtered_image

def edge_detection(image_in, seed_x, seed_y):
    print("Detecting edges from: x:", seed_x, " and y:", seed_y)
    image_in = image_in.astype(np.float32)
    edge_detected_array = itk.GetArrayViewFromImage(level_set.level_set(image_in, seed_x, seed_y)).astype(np.uint8)
    print("Edges detected.")
    return edge_detected_array
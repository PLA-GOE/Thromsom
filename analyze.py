import colorsys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from find_shortest import plot_line, find_shortest_line


def show_color(image_in):
    image_min = np.amin(image_in)
    show_image = (((image_in - image_min) / (np.amax(image_in) - image_min)) * 255.0).astype(np.uint8)
    # show_image = np.rot90(show_image)
    img = Image.fromarray(show_image, 'RGB')
    img.show()


def show_mono(image_in):
    image_min = np.amin(image_in)
    show_image = (((image_in - image_min) / (np.amax(image_in) - image_min)) * 255.0).astype(np.uint8)
    img = Image.fromarray(show_image, 'L')
    img.show()


def trim_to(input_image_array, lo, hi):
    output_image_array = (input_image_array - np.amin(input_image_array)) / float((np.amax(input_image_array) - np.amin(input_image_array))) * (hi - lo) + lo
    return output_image_array


def collapse(out_image_array, blown_up_array):
    print("Collapse, in:", out_image_array.shape, " -- ", blown_up_array.shape)
    # collapses the vessel of the average slice to 1 px
    collapsed_col = np.zeros([out_image_array.shape[0], out_image_array.shape[1], 3], dtype=np.uint8)
    collapsed_bin = np.zeros_like(blown_up_array)
    collapsed_points = []
    for y in range(0, out_image_array.shape[1]):
        print(y)
        x_start = -1
        x_end = -1
        for x in range(0, out_image_array.shape[0]):
            if out_image_array[y, x] != 0:
                collapsed_col[y, x] = [blown_up_array[y, x], blown_up_array[y, x], blown_up_array[y, x]]
                if x_start == -1:
                    x_start = x
            if out_image_array[y, x] == 0:
                collapsed_col[y, x] = [blown_up_array[y, x], blown_up_array[y, x], blown_up_array[y, x]]
                if x_start != -1 and x_end == -1:
                    x_end = (x - 1)
                    # print(y, ":", x_start, "-", x_end)
                    x_val = int(round((x_start + ((x_end - x_start) / 2))))
                    collapsed_col[y, x_val] = [255, 0, 255]
                    collapsed_bin[y, x_val] = 255.0
                    collapsed_points.append((y, x_val))
                    collapsed_col[y, x_start] = [0, 0, 255]
                    collapsed_col[y, x_end] = [0, 0, 255]

    img = Image.fromarray(collapsed_col, 'RGB')
    img.show()

    img = Image.fromarray(collapsed_bin, 'L')
    img.show()

    return collapsed_col, collapsed_bin, collapsed_points


def stats(input_image_array, collapsed_col, collapsed_points):
    pad = 50
    dist_vals = np.zeros((len(collapsed_col[0])), )
    oa_min_dist = 100000
    oa_min_from = -1
    oa_min_to = -1
    oa_max_dist = -1
    oa_max_from = -1
    oa_max_to = -1
    oa_min_cp = [-1, -1]
    oa_max_cp = [-1, -1]

    print("point_ID, v_diameter,x_cp, y_cp")
    for point in range(0, len(collapsed_points)):
        if collapsed_points[point][0] < len(collapsed_col) - pad and collapsed_points[point][1] < len(collapsed_col[0]) - pad and collapsed_points[point][0] > pad and collapsed_points[point][1] > pad:
            # print("=======================")
            # print(collapsed_points[point])
            min_dist, min_from, min_to, angle = find_shortest_line(collapsed_points[point], input_image_array, 0, 180, 1)
            # min_from, min_to, angle = find_shortest_line(collapsed_points[point], out_image_array,angle-18,angle+18,0.01)
            print(point, ",", min_dist, ",", collapsed_points[point][1], ",", collapsed_points[point][0])
            dist_vals[point] = min_dist
            # line_pixels = plot_line(min_from[0], min_from[1], min_to[0], min_to[1])
            # for line_point in line_pixels:
            # collapsed_col[line_point[0], line_point[1]] = [0, 0, 255]
            if min_dist < oa_min_dist:
                oa_min_dist = min_dist
                oa_min_from = min_from
                oa_min_to = min_to
                oa_min_cp = collapsed_points[point]
            if min_dist > oa_max_dist:
                oa_max_dist = min_dist
                oa_max_from = min_from
                oa_max_to = min_to
                oa_max_cp = collapsed_points[point]

    line_pixels = plot_line(oa_min_from[0], oa_min_from[1], oa_min_to[0], oa_min_to[1])
    for line_point in line_pixels:
        collapsed_col[line_point[0], line_point[1]] = [255, 0, 0]

    line_pixels = plot_line(oa_min_cp[0], 0, oa_min_cp[0], oa_min_cp[1])
    for line_point in line_pixels:
        collapsed_col[line_point[0], line_point[1]] = [255, 0, 0]

    line_pixels = plot_line(oa_max_cp[0], 0, oa_max_cp[0], oa_max_cp[1])
    for line_point in line_pixels:
        collapsed_col[line_point[0], line_point[1]] = [0, 255, 0]

    line_pixels = plot_line(oa_max_from[0], oa_max_from[1], oa_max_to[0], oa_max_to[1])
    for line_point in line_pixels:
        collapsed_col[line_point[0], line_point[1]] = [0, 255, 0]

    for x in range(0, dist_vals.size):
        if dist_vals[x] != 0:
            dist_val = (dist_vals[x] - oa_min_dist) / (oa_max_dist - oa_min_dist)
            dist_val_fact = int((dist_vals[x] - oa_min_dist) / (oa_max_dist - oa_min_dist) * 100.0)
            color = colorsys.hsv_to_rgb(dist_val * 0.3, 1, 1)
            collapsed_col[x + pad // 2 + 7, dist_val_fact] = [int(color[0] * 255.0), int(color[1] * 255.0), int(color[2] * 255.0)]

    return collapsed_col
    # collapsed_col = np.einsum('ab->ba', collapsed_col)
    # img = Image.fromarray(collapsed_col, 'RGB')

    # draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype("Ubuntu-B.ttf", 16)
    # draw.text((x, y),"Sample Text",(r,g,b))
    # draw.text((oa_max_to[1] + 100, oa_max_to[0]), ("Max-width: " + str(round(oa_max_dist, 4))), (0, 255, 0), font=font)
    # draw.text((oa_min_to[1] + 100, oa_min_to[0]), ("Min-width: " + str(round(oa_min_dist, 4))), (255, 0, 0), font=font)

    # img.show()

    # scan_fdata = np.einsum('abc->bac', scan_fdata)
    # scan_new_data = np.zeros_like(scan_fdata)
    # scan_new_label = np.zeros_like(scan_fdata)


def trim_thresh(colored_array, new_scan_data, blown_up_array, th_array, id_array, regio_lock):
    print(colored_array.shape)
    print(new_scan_data.shape)
    print(blown_up_array.shape)
    for x in range(regio_lock[0][0], regio_lock[0][1]):
        print("TrimTH:", x)
        for y in range(regio_lock[1][0], regio_lock[1][1]):
            for z in range(0, len(colored_array[x][y])):
                if blown_up_array[x][y] == 1:
                    for i in range(0, len(th_array)):
                        if colored_array[x][y][z] == 0:
                            if th_array[i][0] <= new_scan_data[x][y][z] <= th_array[i][1]:
                                colored_array[x][y][z] = id_array[i]
    return colored_array


def get_area(slice, track_val, id_array, id, x, y):
    sum = 1
    x_sum = x
    y_sum = y
    id_array[x, y] = id
    if x > 0 and y > 0 and x < id_array.shape[0]-1 and y < id_array.shape[1]-1:
        if slice[x + 1, y + 1] == track_val and id_array[x + 1, y + 1] == 0:
            sum_add, x_add, y_add = get_area(slice, track_val, id_array, id, x + 1, y + 1)
            sum += sum_add
            x_sum += x_add
            y_sum += y_add
        if slice[x + 1, y] == track_val and id_array[x + 1, y] == 0:
            sum_add, x_add, y_add = get_area(slice, track_val, id_array, id, x + 1, y)
            sum += sum_add
            x_sum += x_add
            y_sum += y_add
        if slice[x + 1, y - 1] == track_val and id_array[x + 1, y - 1] == 0:
            sum_add, x_add, y_add = get_area(slice, track_val, id_array, id, x + 1, y - 1)
            sum += sum_add
            x_sum += x_add
            y_sum += y_add
        if slice[x, y + 1] == track_val and id_array[x, y + 1] == 0:
            sum_add, x_add, y_add = get_area(slice, track_val, id_array, id, x, y + 1)
            sum += sum_add
            x_sum += x_add
            y_sum += y_add
        if slice[x, y - 1] == track_val and id_array[x, y - 1] == 0:
            sum_add, x_add, y_add = get_area(slice, track_val, id_array, id, x, y - 1)
            sum += sum_add
            x_sum += x_add
            y_sum += y_add
        if slice[x + 1, y + 1] == track_val and id_array[x + 1, y + 1] == 0:
            sum_add, x_add, y_add = get_area(slice, track_val, id_array, id, x + 1, y + 1)
            sum += sum_add
            x_sum += x_add
            y_sum += y_add
        if slice[x + 1, y] == track_val and id_array[x + 1, y] == 0:
            sum_add, x_add, y_add = get_area(slice, track_val, id_array, id, x + 1, y)
            sum += sum_add
            x_sum += x_add
            y_sum += y_add
        if slice[x + 1, y - 1] == track_val and id_array[x + 1, y - 1] == 0:
            sum_add, x_add, y_add = get_area(slice, track_val, id_array, id, x + 1, y - 1)
            sum += sum_add
            x_sum += x_add
            y_sum += y_add
    return sum, x_sum, y_sum


def analyze_id(sid, id, colored_array, path):
    for slice_index in range(0, colored_array.shape[2]):
        current_slice = colored_array[:, :, slice_index]
        thrombus_list = np.where(current_slice == id)
        thrombus_count = len(thrombus_list[0])
        # print("Found:",thrombus_count," thrombus voxels")
        if thrombus_count > 0:
            edge_voxel = 0
            sides = 0
            max_x = np.amax(thrombus_list[0])
            min_x = np.amin(thrombus_list[0])
            max_y = np.amax(thrombus_list[1])
            min_y = np.amin(thrombus_list[1])

            for thrombus_voxel_index in range(0, len(thrombus_list[0])):
                neighbor_missing = 0
                if thrombus_list[0][thrombus_voxel_index] - 1 > 0 and thrombus_list[1][thrombus_voxel_index] - 1 > 0 and thrombus_list[0][thrombus_voxel_index] + 1 < current_slice.shape[0] and thrombus_list[1][thrombus_voxel_index] + 1 < current_slice.shape[1]:
                    if current_slice[thrombus_list[0][thrombus_voxel_index] + 1, thrombus_list[1][thrombus_voxel_index] + 1] == id:
                        neighbor_missing += 1
                    if current_slice[thrombus_list[0][thrombus_voxel_index] + 1, thrombus_list[1][thrombus_voxel_index]] == id:
                        neighbor_missing += 1
                    if current_slice[thrombus_list[0][thrombus_voxel_index] + 1, thrombus_list[1][thrombus_voxel_index] - 1] == id:
                        neighbor_missing += 1
                    if current_slice[thrombus_list[0][thrombus_voxel_index], thrombus_list[1][thrombus_voxel_index] + 1] == id:
                        neighbor_missing += 1
                    if current_slice[thrombus_list[0][thrombus_voxel_index], thrombus_list[1][thrombus_voxel_index] - 1] == id:
                        neighbor_missing += 1
                    if current_slice[thrombus_list[0][thrombus_voxel_index] - 1, thrombus_list[1][thrombus_voxel_index] + 1] == id:
                        neighbor_missing += 1
                    if current_slice[thrombus_list[0][thrombus_voxel_index] - 1, thrombus_list[1][thrombus_voxel_index]] == id:
                        neighbor_missing += 1
                    if current_slice[thrombus_list[0][thrombus_voxel_index] - 1, thrombus_list[1][thrombus_voxel_index] - 1] == id:
                        neighbor_missing += 1
                if neighbor_missing != 8:
                    # print("Edge-Voxel:")
                    edge_voxel += 1
                    sides += (8 - neighbor_missing)
            print("(",sid,",",slice_index, ",", id, ",", path[slice_index][1], ",", path[slice_index][0], ",", min_y, ",", max_y, ",", min_x, ",", max_x, ",", thrombus_count, ",", edge_voxel, ",", sides,"),")


def find_id(tracking_val, colored_array, collapsed_col, rgb, regio_lock, mode):
    print("Tracking:", tracking_val, ", col_ar_shape:", colored_array.shape, "color:", rgb, ", rL:", regio_lock, ", mode:", mode)
    path = np.zeros((colored_array.shape[2], 2))
    # for slice_index in range(0, 4):
    show_mono(colored_array[:, :, 3])
    # show_mono(colored_array[:,:,colored_array.shape[2]-3])
    thresh = 5
    if mode == "biggest":
        print("biggest")
        for slice_index in range(0, colored_array.shape[2]):
            slice = colored_array[:, :, slice_index]
            id_array = np.zeros_like(slice)
            group_list = []
            print(slice_index, "Grouping")
            id = 1
            max_sum = -1
            max_index = -1
            print(colored_array.shape, "--", id_array.shape)
            for x in range(regio_lock[0][0], regio_lock[0][1]):
                for y in range(regio_lock[1][0], regio_lock[1][1]):
                    if slice[x, y] == tracking_val and id_array[x, y] == 0:
                        entry = get_area(slice, tracking_val, id_array, id, x, y)
                        group_list.append(entry)
                        print(id, "Vol:", entry[0], "@", str(entry[1] / entry[0]), "/", str(entry[2] / entry[0]))
                        if entry[0] > max_sum and entry[0] > thresh:
                            max_index = id
                            max_sum = entry[0]
                        id += 1
            print("Max_volume:", max_sum, "with id", max_index)
            if max_sum > 0 and max_index != -1:
                max_entry = group_list[(max_index - 1)]
                print(max_entry[0], ":", max_entry[1], ":", max_entry[2])
                print(round(max_entry[1] / max_entry[0]), ":", round(max_entry[2] / max_entry[0]))
                path[slice_index] = [round(max_entry[1] / max_entry[0]), round(max_entry[2] / max_entry[0])]
            # img = Image.fromarray(thresh, 'L')
            # img.show()
            # img = Image.fromarray(color_output, 'RGB')
            # img.show()ö
        print(path)
        print("Points:", path.shape[0])
        if stats and collapse:
            for stent_point in range(0, (path.shape[0] - 2)):
                if path[stent_point][0] != 0 and path[stent_point][1] != 0 and path[stent_point + 1][0] != 0 and path[stent_point + 1][1] != 0:
                    print(path[stent_point][0], ":", path[stent_point][1])
                    line_pixels = plot_line(int(round(path[stent_point][0])), int(round(path[stent_point][1])), int(round(path[stent_point + 1][0])), int(round(path[stent_point + 1][1])))
                    for line_point in line_pixels:
                        collapsed_col[line_point[0], line_point[1]] = rgb
        return collapsed_col, path
    else:
        for slice_index in range(0, colored_array.shape[2]):
            slice = colored_array[:, :, slice_index]
            id_array = np.zeros_like(slice)
            group_list = []
            print(slice_index, "Grouping")
            id = 1
            max_sum = -1
            max_index = -1
            print(colored_array.shape, "--", id_array.shape)
            for x in range(regio_lock[0][0], regio_lock[0][1]):
                for y in range(regio_lock[1][0], regio_lock[1][1]):
                    if slice[x, y] == tracking_val and id_array[x, y] == 0:
                        entry = get_area(slice, tracking_val, id_array, id, x, y)
                        group_list.append(entry)
                        print(id, "Vol:", entry[0], "@", str(entry[1] / entry[0]), "/", str(entry[2] / entry[0]))
                        if entry[0] > max_sum:
                            if slice_index < colored_array.shape[2] / 2 and entry[2] / entry[0] < colored_array.shape[1]:
                                max_index = id
                                max_sum = entry[0]
                            else:
                                max_index = id
                                max_sum = entry[0]
                        id += 1

            print("Max_volume:", max_sum, "with id", max_index)
            if max_sum != -1 and max_index != -1:
                max_entry = group_list[(max_index - 1)]
                path[slice_index] = (max_entry[1] // max_entry[0], max_entry[2] // max_entry[0])
            # img = Image.fromarray(thresh, 'L')
            # img.show()
            # img = Image.fromarray(color_output, 'RGB')
            # img.show()
        print(path)
        print("Points:", path.shape[0])
        if stats and collapse:
            for stent_point in range(0, (path.shape[0] - 2)):
                line_pixels = plot_line(int(round(path[stent_point][0])), int(round(path[stent_point][1])), int(round(path[stent_point + 1][0])), int(round(path[stent_point + 1][1])))
                for line_point in line_pixels:
                    collapsed_col[line_point[0], line_point[1]] = rgb
        return collapsed_col


def find_area(tracking_id, colored_array):
    print('"Slice","Span-area","min_x","max_x","min_y","max_y"')
    for slice_index in range(colored_array.shape[2]):
        area_arrays = []
        id_array = np.where(colored_array == tracking_id)
        slice = colored_array[:, :, slice_index]
        visited_array = np.zeros_like(slice)
        id = 1
        min_x = 4096
        max_x = -1
        min_y = 4096
        max_y = -1
        for point_index in range(len(id_array[0])):
            if slice[id_array[0][point_index], id_array[1][point_index]] == tracking_id and visited_array[id_array[0][point_index], id_array[1][point_index]] == 0:
                area = get_area(slice, tracking_id, visited_array, id, id_array[0][point_index], id_array[1][point_index])
                area_arrays.append(area)
                id += 1
        # print("Slice-area: ",slice_index)
        stent_area = 0

        for area_index in range(len(area_arrays) - 2):
            ax = area_arrays[area_index][1] // area_arrays[area_index][0]
            ay = area_arrays[area_index][2] // area_arrays[area_index][0]
            if ax > max_x:
                max_x = ax
            if ax < min_x:
                min_x = ax
            if ay > max_y:
                max_y = ay
            if ay < min_y:
                min_y = ay

            bx = area_arrays[area_index + 1][1] // area_arrays[area_index + 1][0]
            by = area_arrays[area_index + 1][2] // area_arrays[area_index + 1][0]

            cx = area_arrays[area_index + 2][1] // area_arrays[area_index + 2][0]
            cy = area_arrays[area_index + 2][2] // area_arrays[area_index + 2][0]
            triangle_area = abs((ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) / 2)
            # print("Tri:",area_index,":",triangle_area)
            stent_area += triangle_area
        for area_index in range(len(area_arrays)):
            ax = area_arrays[area_index][1] // area_arrays[area_index][0]
            ay = area_arrays[area_index][2] // area_arrays[area_index][0]
            if ax > max_x:
                max_x = ax
            if ax < min_x:
                min_x = ax
            if ay > max_y:
                max_y = ay
            if ay < min_y:
                min_y = ay
        print(slice_index, ",", stent_area, ",", min_x, ",", max_x, ",", min_y, ",", max_y)

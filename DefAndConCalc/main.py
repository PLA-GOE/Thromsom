import copy
import math
import nibabel as nb
import numpy as np
from PIL import Image, ImageFont
import colorsys
from PIL import ImageDraw
import os
from scipy.spatial import cKDTree

def find_nifti_files(directory):
    nifti_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.nii', '.nii.gz')):
                abs_path = os.path.abspath(os.path.join(root, file))
                rest_of_path, filename = os.path.split(abs_path)
                if "Output" not in rest_of_path:
                    if "." in filename:
                        name, ext = filename.split('.', 1)  # For handling .nii.gz properly
                        ext = '.' + ext
                    else:
                        name, ext = filename, ''
                    nifti_files.append((rest_of_path, name, ext))

    return nifti_files

dir = "PUT DIR TO SEGMENTATION HERE"

niftis = find_nifti_files(dir)

def hsb_to_rgb(h, s, b):
    r, g, b = colorsys.hsv_to_rgb(h, s, b)
    return int(r * 255), int(g * 255), int(b * 255)
def apply_kernel(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output = np.zeros_like(image)

    # Calculate padding sizes
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image with zeros
    padded_image = np.pad(image, ((pad_width, pad_width), (pad_height, pad_height)), mode='constant', constant_values=0)

    # Convolution operation
    for i in range(pad_width,image_height+pad_width):
        for j in range(pad_height,image_width+pad_height):
            output[i-pad_width, j-pad_height] = np.sum(padded_image[i-pad_width:i + pad_width+1, j-pad_height:j + pad_height+1] * kernel)

    return output

kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

for tuple in niftis:
    out_path = os.path.join(dir,"Output",tuple[1])
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        in_path = os.path.join(tuple[0],tuple[1]+tuple[2])
        print("Working on: ",in_path)
        nb_load = nb.load(in_path)
        nb_fdata = nb_load.get_fdata()
        nb_header = nb_load.header
        nb_affine = nb_load.affine

        print(nb_fdata.shape)

        slices = np.where(nb_fdata==1)
        print(slices[2])
        min_slice = np.amin(slices[2])+1
        max_slice = np.amax(slices[2])
        print(min_slice, max_slice)

        shift = []
        size = []
        centroid = []
        square_centroid = []
        x_prev =  0
        y_prev = 0
        max_x_pos = 0
        max_x_neg = 0
        max_y_pos = 0
        max_y_neg = 0
        #check for thrombi
        for index in range(min_slice, max_slice):
            slice = nb_fdata[:, :, index]
            if np.size(np.where(slice==1)) == 0:
                print("Correction")
                nb_fdata[:,:,index] = nb_fdata[:,:,index-1]
        for index in range(min_slice, max_slice):
            slice = nb_fdata[:,:,index]
            thrombus = np.where(slice==1)
            x_max = np.amax(thrombus[0])
            x_min = np.amin(thrombus[0])
            y_max = np.amax(thrombus[1])
            y_min = np.amin(thrombus[1])
            size_x = x_max-x_min
            size_y = y_max-y_min
            size.append((size_x, size_y))
            sum_avg_x = np.sum(thrombus[0])/len(thrombus[0])
            sum_avg_y = np.sum(thrombus[1])/len(thrombus[1])

            #print(thrombus)
            rounded_x = round(sum_avg_x)
            rounded_y = round(sum_avg_y)
            print(x_min, x_max, y_min, y_max)
            print(size_x, size_y, rounded_x, rounded_y)
            centroid.append((rounded_x, rounded_y))
            if max_x_pos < x_max-rounded_x:
                max_x_pos = x_max-rounded_x
            if max_x_neg < rounded_x-x_min:
                max_x_neg = rounded_x-x_min
            if max_y_pos < y_max-rounded_y:
                max_y_pos = y_max-rounded_y
            if max_y_neg < rounded_y-y_min:
                max_y_neg = rounded_y-y_min
            #print(rounded_x, rounded_y)
            #print(x_min, x_max, y_min, y_max)
            delta_x = x_prev - rounded_x
            delta_y = y_prev - rounded_y
            shift.append((delta_x, delta_y))
            x_prev = rounded_x
            y_prev = rounded_y

        shift.pop(0)
        print(centroid)
        print("Max-Centroid-Shift: MX:",max_x_neg," to ",max_x_pos," and ",max_y_neg," to ",max_y_pos)
        print(shift)
        print(size)
        security_border = 6

        shift_array = np.zeros((max_x_neg+max_x_pos+security_border, max_y_neg+max_y_pos+security_border, max_slice-min_slice))
        print(shift_array.shape)
        slice_0 = shift_array[:, :, 0]

        save_index = 0
        for slice in range(min_slice, max_slice):
            print(save_index)
            print(slice)
            print(centroid[save_index][0]-max_x_neg,"(",centroid[save_index][0],"-,",max_x_neg,")",centroid[save_index][0]+max_x_pos,"(",centroid[save_index][0],"+,",max_x_neg,")",
                  centroid[save_index][1]-max_y_neg,"(",centroid[save_index][1],"-,",max_y_neg,")", centroid[save_index][1]+max_y_pos,"(",centroid[save_index][1],"+,",max_y_pos,")", "S",slice)
            print(nb_fdata[centroid[save_index][0]-max_x_neg:centroid[save_index][0]+max_x_pos,centroid[save_index][1]-max_y_neg:centroid[save_index][1]+max_y_pos,slice].shape)
            shift_array[security_border//2:-security_border//2,security_border//2:-security_border//2,save_index] = nb_fdata[centroid[save_index][0]-max_x_neg:centroid[save_index][0]+max_x_pos,centroid[save_index][1]-max_y_neg:centroid[save_index][1]+max_y_pos,slice]
            save_index+=1

        shift_array[shift_array!=1] = 0

        new_img = nb.Nifti1Image(shift_array, nb_affine, header=nb_header)
        nb.save(new_img, os.path.join(out_path,tuple[1]+"_thrombus.nii"))

        half_border = security_border//2
        wiggle_room_x = list(range(-half_border+1, half_border))
        wiggle_room_y = list(range(-half_border+1, half_border))
        print(wiggle_room_x, wiggle_room_y)
        wiggle_array = []

        wiggle_sum_x = 0
        wiggle_sum_y = 0

        unwiggled_shift_array = np.zeros_like(shift_array)

        unwiggled_shift_array[half_border:-half_border, half_border:-half_border, 0] = shift_array[half_border :-half_border , half_border :-half_border , 0]
        max_wiggle_x = 0
        max_wiggle_y = 0
        min_wiggle_x = shift_array.shape[0]
        min_wiggle_y = shift_array.shape[1]
        for slice_index in range(0, shift_array.shape[2]-1):
            min_sum = np.sum(shift_array.shape[0] * shift_array.shape[1] * shift_array.shape[2])
            #print(min_sum)
            best_x_shift = 0
            best_y_shift = 0
            #print("=============")
            for x_shift in wiggle_room_x:
                for y_shift in wiggle_room_y:
                    start_x = half_border + x_shift
                    end_x = shift_array.shape[0]-half_border + x_shift
                    start_y = half_border + y_shift
                    end_y = shift_array.shape[1]-half_border + y_shift

                    shiftsum = np.sum(np.abs(shift_array[half_border:-half_border, half_border:-half_border,slice_index]-shift_array[start_x:end_x, start_y:end_y,slice_index+1]))
                    #print("SX:",start_x,"EX:", end_x,"SY:", start_y,"EY:", end_y)
                    #print("SS:",shiftsum, x_shift, y_shift)
                    if min_sum > shiftsum:
                        min_sum = shiftsum
                        best_x_shift = x_shift
                        best_y_shift = y_shift
            wiggle_array.append((best_x_shift, best_y_shift))
            #print(wiggle_sum_x,"+",best_x_shift,",", wiggle_sum_y,"+",best_y_shift)
            wiggle_sum_x += best_x_shift
            wiggle_sum_y += best_y_shift
            if wiggle_sum_x > max_wiggle_x:
                max_wiggle_x = wiggle_sum_x
            if wiggle_sum_y > max_wiggle_y:
                max_wiggle_y = wiggle_sum_y

            if wiggle_sum_x < min_wiggle_x:
                min_wiggle_x = wiggle_sum_x
            if wiggle_sum_y < min_wiggle_y:
                min_wiggle_y = wiggle_sum_y

        print("WA:",wiggle_array)
        print("WSX:",wiggle_sum_x,"WSY:", wiggle_sum_y)
        print("MWSX:",max_wiggle_x,"MWSY:", max_wiggle_y)
        abs_wsx = max(abs(max_wiggle_x), abs(min_wiggle_x))
        abs_wsy = max(abs(max_wiggle_y), abs(min_wiggle_y))
        unwiggle_array = np.pad(np.zeros_like(shift_array), pad_width=((abs_wsx, abs_wsx), (abs_wsy, abs_wsy),(0,0)), mode='constant', constant_values=0)
        wiggle_sum_x = 0
        wiggle_sum_y = 0
        print(shift_array.shape)
        print(unwiggle_array.shape)

        uax_mid, uay_mid = unwiggle_array.shape[0]//2, unwiggle_array.shape[1]//2
        sa_ua_start_x, sa_ua_start_y = uax_mid-shift_array.shape[0]//2, uay_mid-shift_array.shape[1]//2
        for slice_index in range(0, shift_array.shape[2]-1):
            print(unwiggle_array.shape)
            print(wiggle_sum_x,"--",wiggle_sum_y)
            print((sa_ua_start_x-wiggle_sum_x),"to",(sa_ua_start_x-wiggle_sum_x+shift_array.shape[0]),",",(sa_ua_start_y-wiggle_sum_y),"to",(sa_ua_start_y-wiggle_sum_y+shift_array.shape[1]))
            print((sa_ua_start_x-wiggle_sum_x+shift_array.shape[0])-(sa_ua_start_x-wiggle_sum_x), "--",(sa_ua_start_y-wiggle_sum_y+shift_array.shape[1])-(sa_ua_start_y-wiggle_sum_y))
            print(shift_array.shape)
            unwiggle_array[(sa_ua_start_x-wiggle_sum_x):(sa_ua_start_x-wiggle_sum_x+shift_array.shape[0]),(sa_ua_start_y-wiggle_sum_y):(sa_ua_start_y-wiggle_sum_y+shift_array.shape[1]),slice_index]=shift_array[:,:,slice_index]
            wiggle_sum_x += wiggle_array[slice_index][0]
            wiggle_sum_y += wiggle_array[slice_index][1]

        new_img = nb.Nifti1Image(unwiggle_array, nb_affine, header=nb_header)
        print(wiggle_array)
        nb.save(new_img, os.path.join(out_path,tuple[1]+"_unwiggle.nii"))

        print(shift_array.shape)
        border_map = np.zeros_like(unwiggle_array)
        border_map_connected = copy.deepcopy(border_map)
        for slice_index in range(0, unwiggle_array.shape[2]):
            border_map[:,:,slice_index] = apply_kernel(unwiggle_array[:,:,slice_index], kernel)

            border_map_connected[border_map>=4] = 1
            border_map_connected[border_map<=3] = 0
            border_map_connected[border_map==0] = 0
            border_map_connected[border_map==9] = 0

        new_img = nb.Nifti1Image(border_map_connected, nb_affine, header=nb_header)
        nb.save(new_img, os.path.join(out_path,tuple[1]+"_borders.nii"))

        img1 = Image.new("RGB", (border_map_connected.shape[0],border_map_connected.shape[1]), "black")
        counter_array = np.zeros((border_map_connected.shape[0],border_map_connected.shape[1],2))
        color_array = np.zeros((border_map_connected.shape[0],border_map_connected.shape[1],3), dtype=np.uint8)
        bmc_len = border_map_connected.shape[2]

        print("Curvature")
        class outline():
            def __init__(self, seed, slice_input, visited_input):
                self.seed = seed
                self.slice_input = slice_input
                print("OL-S:", np.sum(self.slice_input))
                self.visited_input = visited_input
                self.outline_tuples = []
                self.bufferlist = []
                self.bufferlist.append(seed)
                self.get_outline()

            def get_outline(self):
                print("Len-pre:",len(self.outline_tuples))
                print("Vis-pre:",np.sum(self.visited_input))
                while self.bufferlist:
                    value = self.bufferlist.pop()
                    x_local_1, y_local_1 = value[0], value[1]
                    # Define the neighbors' relative positions
                    neighbors = [
                        (x_local_1 + 1, y_local_1 + 1), (x_local_1 + 1, y_local_1), (x_local_1 + 1, y_local_1 - 1),
                        (x_local_1, y_local_1 + 1), (x_local_1, y_local_1 - 1),
                        (x_local_1 - 1, y_local_1 + 1), (x_local_1 - 1, y_local_1), (x_local_1 - 1, y_local_1 - 1)
                    ]

                    for nx, ny in neighbors:
                        if 0 <= nx < self.visited_input.shape[0] and 0 <= ny < self.visited_input.shape[1]:
                            if self.visited_input[nx, ny] == 0 and self.slice_input[nx, ny] == 1:
                                self.bufferlist.append((nx, ny))
                                self.visited_input[nx, ny] = 1
                                self.outline_tuples.append((nx, ny))

                print("Vis-post",np.sum(self.visited_input))
                print("Outline complete:",len(self.outline_tuples))

            def update_visited(self):
                return self.visited_input

            def get_outline_tuples(self):
                return self.outline_tuples

            def get_outline_tuples_len(self):
                return len(self.outline_tuples)

            def __str__(self):
                return "Curve from: ("+str(self.seed[0])+", "+str(self.seed[1])+") with "+str(len(self.outline_tuples))+" pixels:\n"+', '.join([f"({x}, {y})" for x, y in self.outline_tuples])

        def compare_contours(contour1, contour2):
            # Compute the Hausdorff Distance between the contours
            # distance = directed_hausdorff(contour1, contour2)[0]

            # Example arrays of points

            # Create a KDTree for the second array
            tree = cKDTree(contour1)

            # Query the tree for the nearest neighbor of each point in points1
            distances, _ = tree.query(contour2)
            #print("DISTS:",distances)
            distance = np.average(distances)
            #print("DIST:",distance)
            return distance

        curves = []
        for slice in range(0, bmc_len):
            print("Working on: ",slice)
            slice_curves = []
            single_slice = border_map_connected[:,:,slice]
            visited = np.zeros_like(single_slice)
            for x in range(0, single_slice.shape[0]):
                for y in range(0, single_slice.shape[1]):
                    if single_slice[x,y] == 1 and visited[x,y] == 0:
                        new_outline = outline((x,y),single_slice,visited)
                        slice_curves.append(new_outline)
                        visited = new_outline.update_visited()
            print("Sum:",np.sum(single_slice))
            curves.append(sorted(slice_curves, key=lambda obj: obj.get_outline_tuples_len(), reverse=True))

        #for slice_curves in curves:
            #for curve in slice_curves:
                #print(curve)

        color_map = np.zeros_like(unwiggle_array, dtype=np.uint8)
        print(color_map.shape)
        for slice_index in range(0, bmc_len):
            curves_in_slice = curves[slice_index]
            for curve_index in range(0, len(curves_in_slice)):
                current_curve = curves_in_slice[curve_index]
                for pixel in current_curve.get_outline_tuples():
                    #print("Pixel:",pixel, "curve:",str(curve_index+1))
                    color_map[pixel[0],pixel[1],slice_index] = curve_index+1

        new_img = nb.Nifti1Image(color_map, nb_affine, header=nb_header)
        nb.save(new_img, os.path.join(out_path,tuple[1]+"_color_mask.nii"))

        def angle_between_vectors(v1, v2):
            """Calculate the angle between two vectors in radians."""
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            cosine_angle = dot_product / norms
            return np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        def directional_change_indicator(x, y):
            """
            Calculate an indicator of the amount of directional change of a shape's outline.

            Parameters:
                x (list or np.array): x-coordinates of the points.
                y (list or np.array): y-coordinates of the points.

            Returns:
                float: A numerical value representing the amount of directional change.
            """
            x = np.array(x)
            y = np.array(y)

            total_angle_change = 0
            num_points = len(x)

            for i in range(1, num_points - 1):
                # Vector from point i-1 to point i
                v1 = np.array([x[i] - x[i - 1], y[i] - y[i - 1]])
                # Vector from point i to point i+1
                v2 = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])

                # Calculate the angle between v1 and v2
                angle = angle_between_vectors(v1, v2)
                total_angle_change += abs(angle)

            return total_angle_change

        def curvature(x, y):
            """
            Calculate the curvature of a curve defined by points (x, y).

            Parameters:
                x (list or np.array): x-coordinates of the points.
                y (list or np.array): y-coordinates of the points.

            Returns:
                float: A numerical value representing the curvature.
            """
            x = np.array(x)
            y = np.array(y)

            # Calculate first derivatives
            dx = np.gradient(x)
            dy = np.gradient(y)

            # Calculate second derivatives
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)

            # Calculate curvature
            curvature = np.abs(ddx * dy - dx * ddy) / (dx ** 2 + dy ** 2) ** 1.5

            # Return the mean curvature as a representative value
            return np.mean(curvature)
        print("Color done")
        with open(os.path.join(out_path,tuple[1]+"_data.csv"), "a") as file:
            file.write(f"\"slice_index\",\"conture_change(n+1)\",\"curvature\",\"irregularity\"\n")
            for slice_index in range(0, bmc_len-2):
                biggest_curve = np.array(curves[slice_index][0].get_outline_tuples())
                next_curve = np.array(curves[slice_index+1][0].get_outline_tuples())
                x, y = zip(*biggest_curve)
                outline_curvature = curvature(x, y)
                outline_dir_chance = directional_change_indicator(x, y)
                #print(biggest_curve, next_curve)
                file.write(f"\"{slice_index}\",\"{compare_contours(biggest_curve, next_curve)}\",\"{outline_curvature}\",\"{outline_dir_chance}\"\n")
                print(slice_index,":",compare_contours(biggest_curve, next_curve),"-", outline_curvature, "-",outline_dir_chance)

            file.flush()
            file.close()

        for slice in range(0, bmc_len):
            slice_ones = np.where(border_map_connected[:,:,slice]==1)
            print(slice_ones)
            for pixel_index in range(0, len(slice_ones[0])):
                #print(slice_ones[0][pixel_index],slice_ones[1][pixel_index])
                counter_array[slice_ones[0][pixel_index],slice_ones[1][pixel_index],0] = slice
                counter_array[slice_ones[0][pixel_index],slice_ones[1][pixel_index],1] = counter_array[slice_ones[0][pixel_index],slice_ones[1][pixel_index],1]+1

        max_stack = np.amax(counter_array[:,:,1])
        for x in range(0, counter_array.shape[0]):
            for y in range(0, counter_array.shape[1]):
                color_array[x,y] = hsb_to_rgb(counter_array[x,y,0]/bmc_len*0.3, 1.0, 0.5+((counter_array[x,y,1]/max_stack)/2)) if counter_array[x,y,1]!=0 else (255,255,255)

        color_array = np.rot90(color_array, k=1)
        color_array = np.flipud(color_array)
        scale = np.full((color_array.shape[0],100,3), (255, 255, 255), dtype=np.uint8)
        scale[19:(color_array.shape[0]-19),29:41] = (0,0,0)
        scale[19:(color_array.shape[0]-19),69:81] = (0,0,0)
        for x in range(20, (color_array.shape[0]-20)):
            scale[x,30:40] = hsb_to_rgb(0.3-((x/(color_array.shape[0]-20))*0.3), 1.0, 1.0)
            scale[x,70:80] = hsb_to_rgb(1.0, 0.0, 1.0-(((x/(color_array.shape[0]-20))/2)))
        color_array = np.hstack((color_array,scale))

        image = Image.fromarray(color_array, 'RGB')
        draw = ImageDraw.Draw(image)
        # Define the text and font
        font_size = 12
        font = ImageFont.load_default()
        text_color = (0, 0, 0)
        draw.text((counter_array.shape[0]+35, 8), "time", fill=text_color, font=font, anchor="mm")
        draw.text((counter_array.shape[0]+27, 22), str(border_map_connected.shape[2]), fill=text_color, font=font, anchor="rm")
        draw.text((counter_array.shape[0]+27, counter_array.shape[1]-22), "0", fill=text_color, font=font, anchor="rm")
        draw.text((counter_array.shape[0]+75, 8), "stack", fill=text_color, font=font, anchor="mm")
        draw.text((counter_array.shape[0]+67, 22), str(int(max_stack)), fill=text_color, font=font, anchor="rm")
        draw.text((counter_array.shape[0]+67, counter_array.shape[1]-22), "0", fill=text_color, font=font, anchor="rm")

        #image.show()  # Display the image
        image.save(os.path.join(out_path,tuple[1]+"_eval.png"))  # Save the image to a file

        def calculate_distance(pixel1, pixel2):
            """
            Calculate the Euclidean distance between two pixels in a 2D array.

            Parameters:
            pixel1 (tuple): Coordinates of the first pixel (x1, y1).
            pixel2 (tuple): Coordinates of the second pixel (x2, y2).

            Returns:
            float: The Euclidean distance between the two pixels.
            """
            x1, y1 = pixel1
            x2, y2 = pixel2

            delta_x = x2 - x1
            delta_y = y2 - y1

            # Calculate the angle in radians
            angle_radians = math.atan2(delta_y, delta_x)

            # Convert the angle to degrees
            angle_degrees = math.degrees(angle_radians)

            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return distance, angle_degrees

        def calculate_distance_2px(p1, p2):
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            return dist, angle

        min_dist_tuple_index = []
        for slice_index in range(0, border_map_connected.shape[2] - 1):
            current_slice = border_map_connected[:, :, slice_index]
            next_slice = border_map_connected[:, :, slice_index + 1]
            current_pxl = np.column_stack(np.where(current_slice == 1))
            next_pxl = np.column_stack(np.where(next_slice == 1))

            print(slice_index)
            print(len(current_pxl))

            min_dist_tuple_sublist = []

            if current_pxl.size != 0 and next_pxl.size != 0:
                # Compute all pairwise distances
                diff = current_pxl[:, np.newaxis, :] - next_pxl[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)

                # Find minimum distances and their indices
                min_indices = np.argmin(dists, axis=1)
                min_dists = dists[np.arange(len(current_pxl)), min_indices]

                for i, (dist, min_idx) in enumerate(zip(min_dists, min_indices)):
                    if dist == 0:
                        min_dist = 0
                        p2 = next_pxl[min_idx]
                        near_angle = np.arctan2(p2[1] - current_pxl[i, 1], p2[0] - current_pxl[i, 0])
                        min_dist_tuple_sublist.append((min_dist, near_angle, (current_pxl[i, 0], current_pxl[i, 1]), (p2[0], p2[1]), slice_index))
                    else:
                        min_dist = dist
                        p2 = next_pxl[min_idx]
                        near_angle = np.arctan2(p2[1] - current_pxl[i, 1], p2[0] - current_pxl[i, 0])
                        min_dist_tuple_sublist.append((min_dist, near_angle, (current_pxl[i, 0], current_pxl[i, 1]), (p2[0], p2[1]), slice_index))

            min_dist_tuple_index.append(min_dist_tuple_sublist)

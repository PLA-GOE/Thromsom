import matplotlib.pyplot as plt
import numpy as np
import nibabel as nb
import colorsys
import pandas as pd

import os

from PIL import Image
from matplotlib import gridspec

data_path = "PATH TO data.csv FROM DEF AND CON CALC"
mask_path = "PATH TO mask.nii FROM SEGMENTATION"
scan_path = "PATH TO scan.nii (CONVERTED SCAN FROM ORIGINAL DICOM)"

def calculate_hsv(orig, angle, mode):
    if mode == 0:
        if angle == 1:
            hue = 0.0
            saturation = 1.0
            value = orig
        elif angle == 2:
            hue = 0.55
            saturation = 0.7
            value = 1.0
        elif angle == 3.0:
            hue = 0
            saturation = 0.0
            value = 1.0
        else:
            hue = 0
            saturation = 0.0
            value = 1.0


        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)

        return red, green, blue
    elif mode == 1:

        hue = 0
        saturation = 0.0
        value = orig

        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)

        return red, green, blue

image_array = []
def read_niftis(mask_path, scan_path):
    # You can adjust the windowing here
    scan_data = nb.load(scan_path).get_fdata().astype(np.int16)[:,:,:]
    mask_data = nb.load(mask_path).get_fdata()[:,:,:]
    min_mask = np.amin(mask_data)
    max_mask = np.amax(mask_data)
    print(min_mask, max_mask)
    min_scan = np.amin(scan_data)
    max_scan = np.amax(scan_data)
    scan_data = (scan_data-min_scan)/(max_scan-min_scan)
    print(min_scan, max_scan)
    print(np.amin(scan_data), np.amax(scan_data))
    color_array = np.zeros((mask_data.shape[0],mask_data.shape[1],mask_data.shape[2],3))
    print(scan_data.shape)
    print(color_array.shape)
    for z in range(1,scan_data.shape[2]):
        print("Calc:",z)
        for x in range(0, scan_data.shape[0]):
            for y in range(0, scan_data.shape[1]):
                if mask_data[x,y,z] != 0 and mask_data[x,y,z] != 3:
                    color_array[x,y,z] = calculate_hsv(scan_data[x,y,z], mask_data[x,y,z], 0)
                else:
                    color_array[x,y,z] = calculate_hsv(scan_data[x, y, z], mask_data[x, y, z], 1)
        print("Save:",z)
        image = color_array[:, :, z,:]  # Extract the i-th image from the array
        print(image.shape)
        image = image * 255.0

        # Convert the numpy array to a PIL image
        pil_image = Image.fromarray(image.astype(np.uint8))
        image_array.append(pil_image)
        # Save the image
        pil_image.save(f'image_output/image_{z}.png')

        # Read the CSV file
df = pd.read_csv(data_path, header = 0)

# Display the dataframe

def numerical_sort(directory):
    # Extract the numeric part from the directory name
    numeric_part = int(directory.split('_')[1].split('.')[0])
    return numeric_part

print(df)
read = False
if read:
    read_niftis(mask_path, scan_path)
else:
    # Loop through all the files in the folder
    dir_call = os.listdir("image_output")
    directory = sorted(dir_call, key=numerical_sort)
    for filename in directory:
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Read the image and append it to the array
            image = Image.open(os.path.join("image_output", filename))
            image_array.append(image)


def generate_graphs(image_array, df):
    num_steps = 75
    #num_steps = len(image_array)
    graph_width = 3200 // 2  # Width of each graph (2 graphs horizontally)
    graph_height = int(graph_width * 3 / 4)  # Height of each graph (16:9 aspect ratio)

    combined_width = graph_width * 2
    combined_height = graph_height * 2

    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    thr_count_sum = [0]
    print(thr_count_sum[0])
    for val in range(1, 75 - 1):
        thr_count_sum.append(thr_count_sum[val-1]+abs(df['thr_area'][1] - df['thr_area'][val]))
    print(thr_count_sum)
    thr_count_sum = np.asarray(thr_count_sum, dtype=np.float64)
    thr_count_sum /= df['thr_area'][1]
    print(np.max(thr_count_sum))
    print(np.min(thr_count_sum))
    print(thr_count_sum)
    for i in range(1,num_steps-1):
        print(i)
        fig = plt.figure(figsize=(graph_width / 100, graph_height / 100))
        gs = gridspec.GridSpec(3, 2)  # 3 rows, 2 columns

        # Row 0, Col 0: Relative Position of Thrombus and Stent
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("Relative Position of Thrombus and Stent")
        ax1.set_xlim(1, 72)
        ax1.set_ylim(0, 1200)
        ax1.plot(df['index'][1:i], df['thr_y_prox'][1:i], color="#aa0000")
        ax1.plot(df['index'][1:i], df['st_y_dist'][1:i], color="#00aaaa")
        ax1.set_ylabel("Y-Coordinate (Pixel)")
        ax1.set_xlabel("Image Number")
        ax1.legend(["Thrombus, proximal end, y", "Stent tip, distal end, y"])
        ax1.invert_yaxis()

        # Row 0, Col 1: Thrombus Length and Width
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("Thrombus Length and Width over Time")
        ax2.set_xlim(1, 72)
        ax2.set_ylim(0, 300)
        ax2.plot(df['index'][1:i], (df['thr_y_prox'][2:i+1] - df['thr_y_dist'][2:i+1]), color="#aa0000")
        ax2.plot(df['index'][1:i], (df['thr_x_dist'][1:i] - df['thr_x_prox'][1:i]), color="#cc4444")
        ax2.set_ylabel("Length/Width in Pixel")
        ax2.set_xlabel("Image Number")
        ax2.legend(["Thrombus length", "Thrombus width"], bbox_to_anchor=(1, 1.0))

        # Row 1, all columns: Image
        ax_img = fig.add_subplot(gs[1, :])
        #ax_img.imshow(image_array[i], aspect=0.7)
        #emb
        ax_img.imshow(image_array[i+3], aspect=0.7)
        ax_img.axis("off")

        # Row 2, Col 0: Absolute Thrombus Deformation
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.set_title("Absolute Thrombus Deformation over Time")
        ax3.set_xlim(1, 72)
        ax3.set_ylim(0, 35)

        ax3.plot(df['index'][1:i], thr_count_sum[1:i], color="#aa0000")
        ax3.set_ylabel("Thrombus Deformation ($\\cdot10^{-3}$)")
        ax3.set_xlabel("Image Number")

        # Row 2, Col 1: Thrombus Contour Change
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.set_title("Thrombus Contour Change over Time")
        ax4.set_xlim(1, 72)
        ax4.set_ylim(0, 5)
        ax4.plot(df['index'][1:i], df['CC'][1:i], color="#aa0000")
        ax4.set_ylabel("Thrombus Contour Change")
        ax4.set_xlabel("Image Number")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.3)
        # Save the combined image
        plt.savefig(f"output_n39/combined_image_{(i):04d}.png", dpi=72)
        plt.close()

    return combined_image


combined_image = generate_graphs(image_array, df)
plt.imshow(combined_image)
plt.show()

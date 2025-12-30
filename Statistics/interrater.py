import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]
    return data, spacing


def extract_label(mask, label):
    return mask == label


def restrict_z(mask, z_start, z_end):
    restricted = np.zeros_like(mask, dtype=bool)
    restricted[:, :, z_start:z_end] = mask[:, :, z_start:z_end]
    return restricted


def dice_coefficient(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    return 2.0 * intersection / (mask1.sum() + mask2.sum() + 1e-8)


def jaccard_index(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-8)


def extract_surface(mask):
    eroded = binary_erosion(mask)
    return mask ^ eroded


def surface_distances(mask1, mask2, spacing):
    surf1 = extract_surface(mask1)
    surf2 = extract_surface(mask2)

    dt1 = distance_transform_edt(~mask1, sampling=spacing)
    dt2 = distance_transform_edt(~mask2, sampling=spacing)

    d12 = dt2[surf1]
    d21 = dt1[surf2]

    return np.concatenate([d12, d21])


def hausdorff_distance(mask1, mask2, spacing):
    return surface_distances(mask1, mask2, spacing).max()


def hausdorff_distance_95(mask1, mask2, spacing):
    return np.percentile(surface_distances(mask1, mask2, spacing), 95)


def average_surface_distance(mask1, mask2, spacing):
    return surface_distances(mask1, mask2, spacing).mean()


def main( label, z_start, z_end):
    # Label 1 is usually the thrombus. 
    # Z_start is the first slice. We recommend using 1 here, since the first slice is usually with bad contrast. 
    # Z_end is recommended to be set to the last slice where either the thrombus still moves or starts to leave the frame.
    
    mask1_path = "INSERT MASK 1A HERE"
    mask2_path = "INSERT MASK 1B HERE"
    mask1, spacing1 = load_nifti(mask1_path)
    mask2, spacing2 = load_nifti(mask2_path)

    if spacing1 != spacing2:
        raise ValueError("Voxel spacings do not match.")

    if mask1.shape != mask2.shape:
        raise ValueError("Mask dimensions do not match.")

    z_max = mask1.shape[2]
    if not (0 <= z_start < z_end <= z_max):
        raise ValueError(f"Invalid z-range: must be within [0, {z_max})")

    mask1_label = extract_label(mask1, label)
    mask2_label = extract_label(mask2, label)

    mask1_label = restrict_z(mask1_label, z_start, z_end)
    mask2_label = restrict_z(mask2_label, z_start, z_end)

    if mask1_label.sum() == 0 or mask2_label.sum() == 0:
        raise ValueError(
            f"Label {label} not present in selected slice range."
        )
    print("DICE")
    dice = dice_coefficient(mask1_label, mask2_label)
    print(f"Dice coefficient        : {dice:.4f}")
    print("Jaccard")
    jaccard = jaccard_index(mask1_label, mask2_label)
    print(f"Jaccard index           : {jaccard:.4f}")
    print("HD95")
    hd95 = hausdorff_distance_95(mask1_label, mask2_label, spacing1)
    print(f"HD95 (mm)               : {hd95:.2f}")
    print("ASD")
    asd = average_surface_distance(mask1_label, mask2_label, spacing1)
    print(f"Average surface dist(mm): {asd:.2f}")

    print(f"Metrics for label {label}, slices {z_start}:{z_end}")
    print("------------------------------------------------")
    print(f"Dice coefficient        : {dice:.4f}")
    print(f"Jaccard index           : {jaccard:.4f}")
    print(f"HD95 (mm)               : {hd95:.2f}")
    print(f"Average surface dist(mm): {asd:.2f}")


if __name__ == "__main__":
    main(1,1,90)

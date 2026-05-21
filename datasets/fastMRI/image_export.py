import os
import glob
import h5py
# import nibabel as nib       # <- no longer needed if you switch to SimpleITK
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------
# Add an import for SimpleITK:
import SimpleITK as sitk
# -----------------------------------------

# Input directory containing raw h5 files
IN_DIR = "/home/l721f/data/fastmri-breast/rawdata/"

# Output directory for NIfTI images
OUT_DIR = "/home/l721f/data/fastmri-breast/images_for_seg"
os.makedirs(OUT_DIR, exist_ok=True)

# Labels file and status mapping
LABELS_XLSX = "/home/l721f/data/fastmri-breast/fastMRI_breast_labels_short.xlsx"
STATUS_TO_FOLDER = {0: "negative", 1: "malignant", 2: "benign"}

# Load lesion labels mapping: patient_name -> lesion_status
labels_df = pd.read_excel(LABELS_XLSX)
lesion_map = dict(zip(labels_df["patient_name"], labels_df["lesion_status"]))

# Find all raw h5 files, e.g., fastMRI_breast_001_1.h5
h5_files = glob.glob(os.path.join(IN_DIR, "*.h5"))
print(f"Found {len(h5_files)} raw h5 files in {IN_DIR}")

# Process each file
for h5_file in tqdm(h5_files, desc="Exporting 4D image NIfTI from raw"):
    base_name = os.path.splitext(os.path.basename(h5_file))[0]
    series_name = base_name  # includes series suffix like _1
    parts = base_name.split("_")
    if parts and parts[-1].isdigit():
        patient_key = "_".join(parts[:-1])
    else:
        patient_key = base_name

    # Map to lesion status and target folder
    if patient_key not in lesion_map:
        print(f"Skipping {series_name}: no lesion status in labels.")
        continue
    lesion_status = int(lesion_map[patient_key])
    folder_name = STATUS_TO_FOLDER.get(lesion_status, "unknown")
    dest_dir = os.path.join(OUT_DIR, folder_name)
    os.makedirs(dest_dir, exist_ok=True)

    # Read image data from raw h5 file: dataset 'temptv' -> (slices, time, x, y)
    with h5py.File(h5_file, 'r', swmr=True) as f:
        image_data = f['temptv'][:]

    image_data = image_data.astype(np.float32)

    # Reorder from (slices, time, y, x) to (time, slices, y, x) for SimpleITK.
    # This is because SimpleITK expects 4D images in (t, z, y, x) order from NumPy.
    # The spatial orientation (z, y, x) is preserved for each time point.
    vol_tzyx = np.transpose(image_data, (1, 0, 2, 3))

    # Compute subtraction image: t2 - t0
    sub_t2_t0 = vol_tzyx[2] - vol_tzyx[0]  # shape: (slices, y, x)

    # Save subtraction only as 3D NIfTI
    sitk_sub = sitk.GetImageFromArray(sub_t2_t0, isVector=False)
    sitk.WriteImage(sitk_sub, os.path.join(dest_dir, f"{series_name}_sub.nii"))

    # Concatenate: [sub(t2-t0), t0, t1, t2, t3] along time axis and save as 4D NIfTI
    vol_with_sub = np.concatenate([sub_t2_t0[np.newaxis, ...], vol_tzyx], axis=0)
    sitk_4d = sitk.GetImageFromArray(vol_with_sub, isVector=False)
    sitk.WriteImage(sitk_4d, os.path.join(dest_dir, f"{series_name}_image_4d.nii"))

print("Conversion complete!")
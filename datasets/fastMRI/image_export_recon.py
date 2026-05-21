import os
import glob
import h5py
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk


# Directory containing reconstructed image h5 files from reconstruction.py
IN_DIR = "/home/l721f/data/fastmri-breast/images"

# Output directory for NIfTI images (use same directory, just add .nii files)
OUT_DIR = "/home/l721f/data/fastmri-breast/recon_images"
os.makedirs(OUT_DIR, exist_ok=True)


# Find all reconstructed image h5 files, e.g., fastMRI_breast_001_image.h5
h5_files = glob.glob(os.path.join(IN_DIR, "*_image.h5"))
print(f"Found {len(h5_files)} reconstructed image h5 files in {IN_DIR}")


# Process each file: export magnitude of all phases as a 4D NIfTI
for h5_file in tqdm(h5_files, desc="Exporting 4D magnitude NIfTI from recon"):
    base_name = os.path.splitext(os.path.basename(h5_file))[0]  # fastMRI_breast_XXX_image
    patient_name = base_name.replace("_image", "")  # fastMRI_breast_XXX

    # Read reconstructed images: expected shape (slices, time, y, x)
    with h5py.File(h5_file, "r", swmr=True) as f:
        img = f["image"][:]

    # Ensure magnitude, keep all phases
    if np.iscomplexobj(img):
        img = np.abs(img)
    img = img.astype(np.float32, copy=False)

    # Reorder from (slices, time, y, x) to (time, slices, y, x) for SimpleITK (t, z, y, x)
    vol_tzyx = np.transpose(img, (1, 0, 2, 3))

    sitk_4d = sitk.GetImageFromArray(vol_tzyx, isVector=False)
    out_path = os.path.join(OUT_DIR, f"{patient_name}_image.nii")
    sitk.WriteImage(sitk_4d, out_path)

    # Also export subtraction: t2 - t0 as 3D NIfTI (z, y, x)
    sub_zyx = (vol_tzyx[2] - vol_tzyx[0]).astype(np.float32, copy=False)
    sitk_sub = sitk.GetImageFromArray(sub_zyx, isVector=False)
    out_path_sub = os.path.join(OUT_DIR, f"{patient_name}_sub_t2_minus_t0.nii")
    sitk.WriteImage(sitk_sub, out_path_sub)

print("Reconstructed image export complete!")



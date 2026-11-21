"""
NIfTI IO helpers shared by the VFM runners.

Adapted from the ``nnUNet2BiomedParse.py`` helper shipped with BiomedParse v2
(https://github.com/microsoft/BiomedParse, Apache-2.0). Only the 3D NIfTI
paths are kept here — the 2D PNG branches of the upstream helper are not part
of this release.
"""

import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np
from tqdm import tqdm


def load_volume(file_path):
    """Load a 3D NIfTI volume, promoting a 2D slice to a single-slice volume."""
    file_path = Path(file_path)
    nii = nib.load(file_path)
    if len(nii.shape) == 2:
        return nib.Nifti1Image(nii.get_fdata()[..., None], nii.affine)
    if len(nii.shape) != 3:
        raise ValueError(f"Expected a 2D/3D image: {file_path}")
    return nii


def save_volume(data, reference, output_path):
    """Save ``data`` in the reference image's geometry as NIfTI."""
    data = np.asarray(data)
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    nib.save(nib.Nifti1Image(data, reference.affine, header), output_path)


def get_base_name(img_file: Path) -> str:
    """Strip the nnU-Net channel suffix from an image file name."""
    if img_file.name.endswith("_0000.nii.gz"):
        return img_file.name.replace("_0000.nii.gz", "")
    return img_file.stem


def process_single_file(args):
    """Convert a single nnU-Net image into the BiomedParse NPZ layout."""
    img_file, output_path, text_prompts = args

    base_name = get_base_name(img_file)

    try:
        nii = load_volume(img_file)
        img_data = nii.get_fdata()
        if not np.isfinite(img_data).all():
            raise ValueError("Image contains non-finite intensities")
        if img_data.min() < 0 or img_data.max() > 255 or 0 < img_data.max() <= 1:
            low, high = np.percentile(img_data, [0.5, 99.5])
            img_data = np.clip((img_data - low) / max(high - low, 1e-6), 0, 1) * 255
        img_data = img_data.astype(np.uint8)

        output_data = {
            "imgs": img_data,
            "text_prompts": text_prompts,
            "affine": nii.affine,
            "header": np.frombuffer(nii.header.binaryblock, dtype=np.uint8),
        }

        output_file = os.path.join(output_path, f"{base_name}.npz")
        np.savez_compressed(output_file, **output_data)

        return f"Successfully processed: {base_name}", True

    except Exception as e:
        return f"Error processing {base_name}: {str(e)}", False


def convert_nnunet_to_biomedparse(
    input_dir,
    output_dir,
    text_prompts: Dict,
    num_processes: Optional[int] = None,
) -> str:
    """
    Convert an nnU-Net dataset (``imagesTr/``) into BiomedParse NPZ files.

    Args:
        input_dir:    nnU-Net dataset root containing ``imagesTr/``.
        output_dir:   Directory that receives ``{case}.npz`` files.
        text_prompts: Mapping of label ID to text prompt.
        num_processes: Worker processes to use (defaults to the CPU count).

    Returns:
        The output directory.
    """
    if num_processes is None:
        num_processes = cpu_count()

    images_dir = Path(input_dir) / "imagesTr"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    image_files = sorted(images_dir.glob("*_0000.nii.gz"))
    if not image_files:
        raise ValueError("No .nii.gz image files found in imagesTr")

    num_processes = min(num_processes, len(image_files))

    print(f"Found {len(image_files)} image files")
    print(f"Using {num_processes} processes for conversion")

    process_args = [(f, str(output_path), text_prompts) for f in image_files]

    processed_count = 0
    if num_processes == 1:
        results = [process_single_file(a) for a in process_args]
    else:
        with Pool(num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(process_single_file, process_args),
                    total=len(image_files),
                    desc="Converting files",
                )
            )

    for message, success in results:
        if success:
            processed_count += 1
        else:
            print(message)

    if processed_count != len(image_files):
        raise RuntimeError(
            f"Conversion failed for {len(image_files) - processed_count} images"
        )

    print(f"\nConversion completed! Processed {processed_count}/{len(image_files)} files.")
    print(f"Output directory: {output_path}")
    return str(output_path)

#!/usr/bin/env python
"""
Standalone BiomedParse v2 inference runner for Search2Adapt (3D).

This script runs inside the BiomedParse conda environment and is invoked as a
subprocess by ``search2adapt.toolbox.tools.BiomedParseTool``:

    python biomedparse_3d_standalone.py \\
        --input_dir    /path/to/npz \\
        --output_dir   /path/to/out \\
        --checkpoint   /path/to/biomedparse_3D.ckpt \\
        --gpus 0,1 \\
        --biomedparse_root /path/to/BiomedParsev2

Input:
    Directory of ``.npz`` files, each containing:
      - ``imgs``: uint8 volume
      - ``text_prompts``: dict {"1": "organ description", "instance_label": 0}
      - ``gts``: optional ground-truth mask

Output:
    - ``<output_dir>/{case}.npz`` (always)
    - ``<output_dir>/nii_predictions/{case}_pred.nii.gz`` (with ``--save_nii``)
"""

import argparse
import multiprocessing as mp
import os
import sys


def _resolve_backend():
    """Import the vendored inference backend from this script's directory."""
    vfm_dir = os.path.dirname(os.path.abspath(__file__))
    if vfm_dir not in sys.path:
        sys.path.insert(0, vfm_dir)
    from biomedparse_backend import inference as bp_inference

    return bp_inference


def main():
    parser = argparse.ArgumentParser(description="BiomedParse standalone inference")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing .npz input files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output predictions",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to the BiomedParse checkpoint",
    )
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs")
    parser.add_argument(
        "--slice_batch_size",
        type=int,
        default=8,
        help="Number of slices per forward pass",
    )
    parser.add_argument(
        "--interpolate_size",
        type=int,
        default=512,
        help="Spatial resolution for inference",
    )
    parser.add_argument(
        "--nms_threshold", type=float, default=0.5, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Object existence score threshold",
    )
    parser.add_argument(
        "--biomedparse_root",
        type=str,
        default="",
        help="Path to the BiomedParse v2 checkout (model code + hydra configs)",
    )
    parser.add_argument(
        "--save_nii",
        action="store_true",
        help="Also save predictions as NIfTI (.nii.gz)",
    )
    parser.add_argument(
        "--save_prob", action="store_true", help="Also save probability maps"
    )
    args = parser.parse_args()

    biomedparse_root = args.biomedparse_root or os.environ.get("BIOMEDPARSE_ROOT", "")
    if biomedparse_root:
        biomedparse_root = os.path.abspath(os.path.expanduser(biomedparse_root))
        # The model definition (src/) and the hydra configs live in the checkout.
        if biomedparse_root not in sys.path:
            sys.path.insert(0, biomedparse_root)

    bp_inference = _resolve_backend()

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    bp_inference.CUDA_DEVICES = gpu_list
    bp_inference.NUM_WORKERS = len(gpu_list)
    bp_inference.INPUT_DIR = args.input_dir
    bp_inference.OUTPUT_DIR = args.output_dir
    bp_inference.CHECKPOINT_PATH = args.checkpoint
    bp_inference.SLICE_BATCH_SIZE = args.slice_batch_size
    bp_inference.INTERPOLATE_SIZE = args.interpolate_size
    bp_inference.NMS_IOU_THRESHOLD = args.nms_threshold
    bp_inference.SCORE_THRESHOLD = args.score_threshold
    bp_inference.SAVE_NII = args.save_nii
    bp_inference.SAVE_PROB = args.save_prob
    bp_inference.NII_OUTPUT_DIR = None  # defaults to output_dir/nii_predictions
    bp_inference.PROB_OUTPUT_DIR = None
    bp_inference.BIOMEDPARSE_ROOT = biomedparse_root

    print("[run_biomedparse_standalone] Config:")
    print(f"  GPUs: {gpu_list}")
    print(f"  Input: {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Batch size: {args.slice_batch_size}")
    print(f"  Interpolate: {args.interpolate_size}")
    print(f"  Save NII: {args.save_nii}")
    print(f"  BiomedParse root: {biomedparse_root or '<unset>'}")

    bp_inference.main()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

#!/usr/bin/env python
"""
Standalone SAM 3 inference runner for Search2Adapt (3D NIfTI).

SAM 3 ships as a video predictor, so a 3D volume is converted into a frame
sequence, prompted per organ, and propagated in both directions along the
slice axis.

This script runs inside the SAM 3 conda environment and is invoked as a
subprocess by ``search2adapt.toolbox.tools.SAM3Tool``:

    python sam3_3d_standalone.py \\
        --json_config sam3_config.json \\
        --output_dir  out/ \\
        --checkpoint  /path/to/sam3.pt \\
        --gpus 0,1 \\
        --sam3_root   /path/to/sam3

Adapted from the standalone runner shipped with BiomedParse v2's ecosystem
(https://github.com/microsoft/BiomedParse, Apache-2.0); the 2D PNG branch of
the original script is not part of this release.
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Optional

import cv2
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

# The vendored helper is imported by path so this script also works when it is
# executed as a file (not as part of an installed package).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnunet_io import load_volume  # noqa: E402


def _resolve_sam3_root(explicit_root: str) -> str:
    if explicit_root:
        return os.path.abspath(os.path.expanduser(explicit_root))
    env_root = os.environ.get("SAM3_ROOT", "")
    if env_root:
        return os.path.abspath(os.path.expanduser(env_root))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3")


class StandaloneSAM3Predictor:
    """Thin wrapper around the SAM 3 video predictor."""

    def __init__(self, checkpoint_path: str, gpus_to_use: List[int]):
        from sam3.model_builder import build_sam3_video_predictor

        print(f"Loading SAM3 model from {checkpoint_path} on GPUs {gpus_to_use}...")
        self.predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use, checkpoint_path=checkpoint_path
        )
        print("Model loaded.")

    @staticmethod
    def _load_and_transpose(nii_path: str, slice_axis: Optional[int] = None):
        """
        Load a NIfTI volume and move the slice axis to the front.

        The transpose must stay consistent with the orientation
        standardisation used by the toolbox, so that the bounding boxes
        handed to SAM 3 refer to the same voxel grid.
        """
        img_obj = load_volume(nii_path)
        data = img_obj.get_fdata()
        affine = img_obj.affine
        header = img_obj.header

        shape = data.shape
        min_dim_idx = int(np.argmin(shape)) if slice_axis is None else int(slice_axis)
        if min_dim_idx not in (0, 1, 2):
            raise ValueError("Invalid slice axis")

        # Target layout: (D, H, W)
        permute_order = [0, 1, 2]
        permute_order.pop(min_dim_idx)
        permute_order.insert(0, min_dim_idx)

        return data.transpose(tuple(permute_order)), affine, header, permute_order

    @staticmethod
    def _nii_to_frames(data_transposed, temp_base_dir, case_id):
        """Write the volume as a JPEG frame sequence and return the frame dir."""
        save_dir = os.path.join(temp_base_dir, case_id)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        lower = np.percentile(data_transposed, 0.5)
        upper = np.percentile(data_transposed, 99.5)
        if (upper - lower) > 1e-5:
            data_norm = (data_transposed - lower) / (upper - lower) * 255.0
        else:
            data_norm = np.zeros_like(data_transposed)

        data_norm = np.clip(data_norm, 0, 255).astype(np.uint8)

        depth, height, width = data_norm.shape
        # SAM3's CUDA antialias kernel cannot downsample to extremely small frames.
        scale = max(1.0, 64 / min(height, width))
        frame_w, frame_h = int(round(width * scale)), int(round(height * scale))

        for i in range(depth):
            slice_rgb = cv2.cvtColor(data_norm[i, :, :], cv2.COLOR_GRAY2BGR)
            if scale > 1:
                slice_rgb = cv2.resize(
                    slice_rgb, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR
                )
            cv2.imwrite(os.path.join(save_dir, f"{i:05d}.jpg"), slice_rgb)

        return save_dir, frame_w, frame_h

    @staticmethod
    def _normalize_box(box, w, h):
        """Normalize a [x, y, bw, bh] box to [0, 1]."""
        x, y, bw, bh = box

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        if bw <= 0 or bh <= 0:
            return None
        return [x / w, y / h, bw / w, bh / h]

    def process_case(
        self,
        case_id,
        image_path,
        prompts: List[Dict],
        output_dir,
        temp_dir,
        slice_axis: Optional[int] = None,
    ):
        """Run SAM 3 for one case and write ``{case_id}.nii.gz`` to ``output_dir``."""
        session_id = None
        vid_dir = None
        try:
            data_T, affine, header, permute_order = self._load_and_transpose(
                image_path, slice_axis
            )
            vid_dir, _, _ = self._nii_to_frames(data_T, temp_dir, case_id)

            num_frames = data_T.shape[0]
            original_h, original_w = data_T.shape[1:]

            final_volume = np.zeros(data_T.shape, dtype=np.uint8)

            for p in prompts:
                # Each organ gets an independent session so that masks of one
                # organ cannot contaminate the next.
                resp = self.predictor.handle_request(
                    {"type": "start_session", "resource_path": vid_dir}
                )
                session_id = resp["session_id"]
                obj_id = p["obj_id"]
                key_slice = p.get("key_slice", 0)

                if key_slice < 0 or key_slice >= num_frames:
                    raise ValueError(
                        f"Invalid key slice {key_slice} for {num_frames} frames"
                    )

                req = {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "obj_id": 1,
                    "frame_index": key_slice,
                }

                if "text" in p:
                    req["text"] = p["text"]

                if "box" in p:
                    norm_box = self._normalize_box(
                        p["box"], original_w, original_h
                    )
                    if norm_box:
                        req["bounding_boxes"] = [norm_box]
                        req["bounding_box_labels"] = [1]

                self.predictor.handle_request(req)

                stream = self.predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "both",
                    }
                )

                for r in stream:
                    idx = r["frame_index"]
                    if not 0 <= idx < num_frames:
                        raise ValueError("SAM3 returned an invalid frame index")
                    masks = r["outputs"]["out_binary_masks"]
                    if len(masks) > 0:
                        if isinstance(masks, torch.Tensor):
                            masks = masks.cpu().numpy()
                        m = np.any(np.asarray(masks), axis=0)

                        if m.shape != (original_h, original_w):
                            m = (
                                cv2.resize(
                                    m.astype(np.uint8),
                                    (original_w, original_h),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                .astype(bool)
                            )

                        final_volume[idx, final_volume[idx] == obj_id] = 0
                        final_volume[idx, (m > 0) & (final_volume[idx] == 0)] = obj_id

                self.predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )
                session_id = None

            inv_order = np.argsort(permute_order)
            final_orig = final_volume.transpose(tuple(inv_order))

            out_name = f"{case_id}.nii.gz"
            os.makedirs(output_dir, exist_ok=True)
            nib.save(
                nib.Nifti1Image(final_orig, affine, header),
                os.path.join(output_dir, out_name),
            )
            print(f"Saved: {out_name}")

        finally:
            if session_id:
                self.predictor.handle_request(
                    {"type": "close_session", "session_id": session_id}
                )
            if vid_dir and os.path.isdir(vid_dir):
                shutil.rmtree(vid_dir)


def main():
    parser = argparse.ArgumentParser(description="SAM 3 standalone inference")
    parser.add_argument(
        "--json_config",
        type=str,
        required=True,
        help="Path to the JSON config with the per-case prompts",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument(
        "--sam3_root",
        type=str,
        default="",
        help="Path to the SAM 3 checkout (defaults to $SAM3_ROOT)",
    )
    args = parser.parse_args()

    sam3_root = _resolve_sam3_root(args.sam3_root)
    if sam3_root not in sys.path:
        sys.path.append(sam3_root)

    checkpoint = args.checkpoint or os.path.join(sam3_root, "sam3", "sam3.pt")
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]

    predictor = StandaloneSAM3Predictor(checkpoint, gpus)

    with open(args.json_config, "r") as f:
        config = json.load(f)

    temp_dir = os.path.join(args.output_dir, "temp_frames")

    try:
        for case_name, info in tqdm(config.items(), desc="Running SAM3"):
            predictor.process_case(
                case_name,
                info["image"],
                info["prompts"],
                args.output_dir,
                temp_dir,
                slice_axis=info.get("slice_axis"),
            )
    finally:
        predictor.predictor.shutdown()


if __name__ == "__main__":
    main()

"""
Search2Adapt Toolbox — Action Space (动作空间)

MONAI-native tool implementations for medical image segmentation.
VFM wrappers (SAM3, BiomedParse) and nnUNet↔BiomedParse data conversion.

Design:
  - ALL tools use monai.transforms exclusively (scipy.ndimage used only by SymmetryCheckerTool, diagnostic)
  - Morphological ops use monai.transforms.dilate/erode (MONAI 1.5.2)
  - Preprocessing tools designed for [0, 255] preprocessed data (not raw HU)
  - VFM wrappers unchanged (SAM3 subprocess, BiomedParse multiprocessing)
  - Unified I/O: NIfTI path in → NIfTI path out (affine/header preserved)
  - Tool selection informed by perceptor memory image_profile features

Tool categories and perceptor feature drivers:
  vfm_proposer:   biomedparse, sam3 (external VFMs, not refactored)
  preprocessing:  enhance_image  → intensity_skewness, histogram_num_peaks
                  smooth_image   → noise_std, snr_estimate, noise_level_auto
  postprocessing: postprocess_segmentation → default cleanup pipeline
                  gaussian_smooth_mask     → jagged boundaries
                  morphological_closing    → holes, broken boundaries
                  morphological_opening    → spurious protrusions
                  label_filter             → remove incorrect labels
  diagnostic:     symmetry_checker         → paired organ validation
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    import torch
    from monai.transforms import (
        AdjustContrast,
        Compose,
        EnsureType,
        FillHoles,
        GaussianSmooth,
        HistogramNormalize,
        KeepLargestConnectedComponent,
        LabelFilter,
        MedianSmooth,
        RemoveSmallObjects,
        ScaleIntensityRange,
        ScaleIntensityRangePercentiles,
        dilate,
        erode,
    )

    TORCH_AVAILABLE = True
    MONAI_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    MONAI_AVAILABLE = False
    KeepLargestConnectedComponent = None
    FillHoles = None
    RemoveSmallObjects = None
    LabelFilter = None
    GaussianSmooth = None
    MedianSmooth = None
    AdjustContrast = None
    HistogramNormalize = None
    ScaleIntensityRange = None
    ScaleIntensityRangePercentiles = None
    dilate = None
    erode = None
    Compose = None
    EnsureType = None


# =============================================================================
#  Tool Registry: text descriptions for Strategist LLM
# =============================================================================

TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "biomedparse": {
        "category": "vfm_proposer",
        "name": "BiomedParse",
        "description": (
            "Text-prompted semantic segmentation VFM (371M params). "
            "Generates semantic masks grounded in anatomical meaning. "
            "Input: image + text prompt. Output: multi-label mask."
        ),
        "applicable_when": (
            "Need to generate initial pseudo-labels from scratch "
            "or re-generate for a specific organ."
        ),
        "params": {
            "slice_batch_size": {"type": "int", "default": 8},
            "interpolate_size": {"type": "int", "default": 512},
        },
    },
    "sam3": {
        "category": "vfm_proposer",
        "name": "SAM3",
        "description": (
            "Geometric segmentation VFM (862M params) using bounding box + text prompts. "
            "Excels at boundary refinement and spatial consistency. "
            "Input: image + bounding box + text. Output: binary/multi-label mask."
        ),
        "applicable_when": (
            "Need to refine boundaries of existing coarse mask, "
            "or re-segment a specific region with visual prompts."
        ),
        "params": {
            "gpus": {
                "type": "list[int]",
                "default": [0],
            },
        },
    },
    "dual_expert": {
        "category": "vfm_proposer",
        "name": "DualExpert",
        "description": (
            "Two-stage VFM pipeline: BiomedParse generates initial pseudo-labels, "
            "then key slices and bounding boxes are extracted per organ, "
            "and SAM3 re-segments using text+box prompts. "
            "Results are fused with BP priority (union strategy). "
            "Combines BiomedParse's semantic understanding with SAM3's boundary precision."
        ),
        "applicable_when": (
            "Want highest quality pseudo-labels by combining two VFMs. "
            "Best for initial pseudo-label generation when both VFMs are available. "
            "More expensive than single-VFM but produces better boundaries."
        ),
        "params": {
            "dice_threshold": {
                "type": "float",
                "default": 0.5,
                "description": (
                    "BP-SAM3 agreement threshold. If per-case Dice >= threshold, "
                    "fuse results (union with BP priority). Otherwise use BP only."
                ),
            },
        },
    },
    # ---- Preprocessing (operates on image, not mask) ----
    "enhance_image": {
        "category": "preprocessing",
        "name": "EnhanceImage",
        "description": (
            "MONAI intensity enhancement for preprocessed [0, 255] images. "
            "Two modes: 'gamma' (AdjustContrast — corrects intensity skew), "
            "'histogram' (HistogramNormalize — equalizes multi-peak distributions). "
            "Improves VFM input quality without changing spatial properties."
        ),
        "applicable_when": (
            "Image has high intensity skewness (>1.5 in image_profile) or "
            "complex histogram (>5 peaks). Check image_profile features: "
            "intensity_skewness, histogram_num_peaks, dynamic_range_utilization."
        ),
        "params": {
            "mode": {
                "type": "str",
                "default": "gamma",
                "description": (
                    "Enhancement mode: 'gamma' (contrast correction), "
                    "'histogram' (histogram equalization)."
                ),
            },
            "gamma": {
                "type": "float",
                "default": 0.7,
                "range": [0.3, 2.0],
                "description": (
                    "Gamma value for 'gamma' mode. <1.0 brightens dark regions "
                    "(use when intensity_skewness > 1.5). >1.0 darkens bright regions."
                ),
            },
            "num_bins": {
                "type": "int",
                "default": 256,
                "description": "Number of bins for 'histogram' mode.",
            },
        },
    },
    "smooth_image": {
        "category": "preprocessing",
        "name": "SmoothImage",
        "description": (
            "MONAI noise reduction for images before VFM inference. "
            "Two methods: 'gaussian' (GaussianSmooth — general denoising) or "
            "'median' (MedianSmooth — edge-preserving denoising). "
            "Reduces noise that confuses VFM boundary detection."
        ),
        "applicable_when": (
            "Image has high noise (noise_level_auto='High' or snr_estimate < 15 "
            "in image_profile). Gaussian for general denoising, median for "
            "edge-preserving denoising near organ boundaries. "
            "Check image_profile features: noise_std, snr_estimate, noise_level_auto."
        ),
        "params": {
            "method": {
                "type": "str",
                "default": "gaussian",
                "description": "'gaussian' or 'median'.",
            },
            "sigma": {
                "type": "float",
                "default": 1.0,
                "range": [0.5, 3.0],
                "description": (
                    "Sigma for gaussian method. Higher = more smoothing. "
                    "Set ~noise_std/10 from image_profile for calibrated denoising."
                ),
            },
            "radius": {
                "type": "int",
                "default": 1,
                "range": [1, 3],
                "description": "Radius for median method.",
            },
        },
    },
    # ---- Post-processing (operates on mask — all MONAI-native) ----
    "postprocess_segmentation": {
        "category": "postprocessing",
        "name": "PostprocessSegmentation",
        "description": (
            "MONAI Compose pipeline for segmentation cleanup: "
            "remove small objects → keep largest connected component → fill holes. "
            "All steps use monai.transforms."
        ),
        "applicable_when": (
            "Mask has scattered noise, small fragments, or holes. "
            "Use this as the default cleanup pipeline."
        ),
        "params": {
            "min_size": {
                "type": "int",
                "default": 64,
                "description": "Minimum voxel count to keep (RemoveSmallObjects).",
            },
            "keep_largest_only": {
                "type": "bool",
                "default": True,
                "description": "If True, keep only the largest connected component per label.",
            },
            "fill_holes": {
                "type": "bool",
                "default": True,
                "description": "If True, fill enclosed holes per label (FillHoles).",
            },
            "connectivity": {
                "type": "int",
                "default": 2,
                "description": "Connectivity for connected component analysis (1-3).",
            },
        },
    },
    "gaussian_smooth_mask": {
        "category": "postprocessing",
        "name": "GaussianSmoothMask",
        "description": (
            "Smooths mask boundaries via MONAI GaussianSmooth on per-label "
            "binary masks (float smooth + re-threshold at 0.5). "
            "Produces anatomically plausible contours."
        ),
        "applicable_when": (
            "Mask boundaries are jagged or staircase-like, need smoother contours."
        ),
        "params": {
            "sigma": {"type": "float", "default": 1.0, "range": [0.5, 3.0]},
        },
    },
    "morphological_closing": {
        "category": "postprocessing",
        "name": "MorphologicalClosing",
        "description": (
            "Morphological closing (dilation → erosion) via MONAI dilate/erode. "
            "Fills small holes and connects nearby fragments."
        ),
        "applicable_when": (
            "Mask has small holes inside organs or broken boundaries with thin gaps."
        ),
        "params": {
            "filter_size": {
                "type": "int",
                "default": 3,
                "range": [3, 7],
                "description": "Size of the morphological structuring element.",
            },
            "iterations": {"type": "int", "default": 1, "range": [1, 3]},
        },
    },
    "morphological_opening": {
        "category": "postprocessing",
        "name": "MorphologicalOpening",
        "description": (
            "Morphological opening (erosion → dilation) via MONAI dilate/erode. "
            "Removes small protrusions and noise while preserving shape."
        ),
        "applicable_when": (
            "Mask has small spurious protrusions or thin spiky artifacts at boundaries."
        ),
        "params": {
            "filter_size": {
                "type": "int",
                "default": 3,
                "range": [3, 7],
                "description": "Size of the morphological structuring element.",
            },
            "iterations": {"type": "int", "default": 1, "range": [1, 3]},
        },
    },
    "label_filter": {
        "category": "postprocessing",
        "name": "LabelFilter",
        "description": (
            "MONAI LabelFilter — removes specific labels from a multi-label mask. "
            "Keeps only the specified labels, sets all others to 0. "
            "Useful for removing labels that the Evaluator identified as incorrect."
        ),
        "applicable_when": (
            "Evaluator reports specific organs have very low quality scores, "
            "or symmetry_checker found abnormal paired organs. "
            "Remove bad labels before re-generating them with VFM."
        ),
        "params": {
            "applied_labels": {
                "type": "list[int]",
                "default": [],
                "description": (
                    "Label IDs to KEEP. All other labels are set to 0. "
                    "Use organ_label_map to map organ names to label IDs."
                ),
            },
        },
    },
    # ---- Diagnostics ----
    "symmetry_checker": {
        "category": "diagnostic",
        "name": "SymmetryChecker",
        "description": (
            "Checks volume ratio and flip-overlap Dice for paired left/right organs. "
            "Returns diagnostic report (does NOT modify mask)."
        ),
        "applicable_when": (
            "Dataset contains paired organs (e.g., Left/Right Kidney) "
            "and one side may be missing or malformed."
        ),
        "params": {
            "organ_pairs": {
                "type": "list[dict]",
                "default": [],
                "description": "[{'Left Kidney': 1, 'Right Kidney': 2}]",
            },
            "vol_threshold": {"type": "float", "default": 0.2},
            "dice_threshold": {"type": "float", "default": 0.1},
        },
    },
    # ---- New postprocessing tools ----
    "normalize_intensity": {
        "category": "preprocessing",
        "name": "NormalizeIntensity",
        "description": (
            "Window/level intensity normalization using MONAI ScaleIntensityRange. "
            "Clips image intensities to [a_min, a_max] and rescales to [b_min, b_max]. "
            "Optionally clips output values. Useful for standardizing CT HU values "
            "or normalizing images with extreme outlier intensities."
        ),
        "applicable_when": (
            "Image has inconsistent intensity ranges across cases, "
            "extreme outliers in intensity distribution, or "
            "dynamic_range_utilization < 0.5 in image_profile. "
            "Particularly useful for CT images with metal artifacts or "
            "when intensity_skewness > 2.0."
        ),
        "params": {
            "a_min": {
                "type": "float",
                "default": -1000.0,
                "description": "Lower input intensity bound (e.g., -1000 for CT air).",
            },
            "a_max": {
                "type": "float",
                "default": 1000.0,
                "description": "Upper input intensity bound (e.g., 1000 for CT bone).",
            },
            "b_min": {
                "type": "float",
                "default": 0.0,
                "description": "Lower output intensity bound.",
            },
            "b_max": {
                "type": "float",
                "default": 255.0,
                "description": "Upper output intensity bound.",
            },
            "clip": {
                "type": "bool",
                "default": True,
                "description": "Clip output values to [b_min, b_max].",
            },
        },
    },
    "clip_intensity": {
        "category": "preprocessing",
        "name": "ClipIntensity",
        "description": (
            "Robust percentile-based intensity clipping using MONAI "
            "ScaleIntensityRangePercentiles. Clips intensities at the given "
            "lower/upper percentile and rescales the result. More robust than "
            "fixed window normalization when intensity distributions vary."
        ),
        "applicable_when": (
            "Image has long-tailed intensity distribution or significant outliers "
            "that compress the visible intensity range. "
            "Check image_profile: if intensity_skewness > 1.5 or "
            "dynamic_range_utilization < 0.6."
        ),
        "params": {
            "lower": {
                "type": "float",
                "default": 1.0,
                "range": [0.0, 10.0],
                "description": "Lower percentile for clipping (e.g., 1 = 1st percentile).",
            },
            "upper": {
                "type": "float",
                "default": 99.0,
                "range": [90.0, 100.0],
                "description": "Upper percentile for clipping.",
            },
            "b_min": {
                "type": "float",
                "default": 0.0,
                "description": "Lower output bound after rescaling.",
            },
            "b_max": {
                "type": "float",
                "default": 255.0,
                "description": "Upper output bound after rescaling.",
            },
        },
    },
    "iterative_hole_fill": {
        "category": "postprocessing",
        "name": "IterativeHoleFill",
        "description": (
            "Repeatedly applies MONAI FillHoles with increasing connectivity "
            "to close progressively larger holes in mask regions. "
            "More aggressive than single-pass FillHoles in postprocess_segmentation. "
            "Fills holes left by standard cleanup for organs with complex topology."
        ),
        "applicable_when": (
            "Mask has medium-to-large holes inside organ regions that single-pass "
            "FillHoles fails to close. Useful for liver, spleen, large solid organs "
            "with internal heterogeneity causing segmentation gaps."
        ),
        "params": {
            "iterations": {
                "type": "int",
                "default": 3,
                "range": [1, 5],
                "description": "Number of FillHoles passes (each increases effective radius).",
            },
            "connectivity": {
                "type": "int",
                "default": 1,
                "range": [1, 3],
                "description": "Connectivity for hole filling (1=face, 2=edge, 3=vertex).",
            },
        },
    },
    "boundary_refinement": {
        "category": "postprocessing",
        "name": "BoundaryRefinement",
        "description": (
            "Sharpens mask boundaries via erosion followed by selective dilation "
            "using MONAI dilate/erode. Removes thin fuzzy boundary layers then "
            "re-expands to recover true organ extent. Improves boundary precision "
            "for organs with diffuse edges caused by partial volume effects."
        ),
        "applicable_when": (
            "Mask boundaries are fuzzy or over-segmented with thin halos around organs. "
            "Useful after smooth_image when boundaries become too diffuse, "
            "or when Evaluator reports low boundary_precision scores."
        ),
        "params": {
            "erode_size": {
                "type": "int",
                "default": 3,
                "range": [3, 5],
                "description": "Structuring element size for initial erosion.",
            },
            "dilate_size": {
                "type": "int",
                "default": 3,
                "range": [3, 5],
                "description": "Structuring element size for subsequent dilation.",
            },
            "iterations": {
                "type": "int",
                "default": 1,
                "range": [1, 2],
                "description": "Number of erosion-dilation cycles.",
            },
        },
    },
    "multi_label_closing": {
        "category": "postprocessing",
        "name": "MultiLabelClosing",
        "description": (
            "Per-label morphological closing with label-specific kernel sizes. "
            "Allows different organs to receive different closing strengths "
            "based on their expected morphology and typical defect patterns. "
            "Small organs (label_sizes dict maps label_id → filter_size)."
        ),
        "applicable_when": (
            "Multi-organ segmentation where different organs have different hole/gap sizes. "
            "E.g., liver needs larger closing (filter_size=7) than kidney (filter_size=3). "
            "Use when morphological_closing with uniform kernel is suboptimal."
        ),
        "params": {
            "default_filter_size": {
                "type": "int",
                "default": 3,
                "range": [3, 7],
                "description": "Default kernel size for labels not in label_sizes.",
            },
            "label_sizes": {
                "type": "dict",
                "default": {},
                "description": (
                    "Dict mapping label_id (int) → filter_size (int). "
                    "E.g., {1: 7, 2: 3} for larger liver (1) vs smaller kidney (2)."
                ),
            },
            "iterations": {"type": "int", "default": 1, "range": [1, 3]},
        },
    },
    "remove_outlier_labels": {
        "category": "postprocessing",
        "name": "RemoveOutlierLabels",
        "description": (
            "Removes segmentation fragments that are spatially far from the "
            "main organ body, using centroid distance filtering. "
            "For each label, keeps only the connected component(s) whose centroid "
            "is within max_distance voxels of the largest component's centroid. "
            "Eliminates anatomically implausible false positive fragments."
        ),
        "applicable_when": (
            "Mask has distant spurious fragments far from the true organ location. "
            "E.g., small liver fragments appearing near the pelvis. "
            "More targeted than postprocess_segmentation min_size filtering "
            "when fragments are large but spatially displaced."
        ),
        "params": {
            "max_distance": {
                "type": "float",
                "default": 50.0,
                "range": [20.0, 200.0],
                "description": (
                    "Maximum allowed distance (voxels) from main component centroid. "
                    "Components beyond this are removed."
                ),
            },
            "min_size": {
                "type": "int",
                "default": 10,
                "description": "Minimum voxel count — components smaller than this are always removed.",
            },
        },
    },
    "conditional_fill": {
        "category": "postprocessing",
        "name": "ConditionalFill",
        "description": (
            "Fills holes in specific labels only when their volume falls below "
            "an expected minimum threshold. Prevents over-aggressive hole filling "
            "on well-segmented organs while targeting underfilled ones. "
            "Uses MONAI FillHoles internally."
        ),
        "applicable_when": (
            "Some organs are well-segmented (don't need hole filling) while others "
            "consistently appear hollow or underfilled. Use when global hole filling "
            "would modify already-correct organs. Specify expected_min_voxels "
            "based on anatomical knowledge (e.g., liver > 50000 voxels in abdominal CT)."
        ),
        "params": {
            "expected_min_voxels": {
                "type": "dict",
                "default": {},
                "description": (
                    "Dict mapping label_id (int) → minimum expected voxel count. "
                    "Fill holes only for labels below their threshold. "
                    "E.g., {1: 50000, 2: 10000} for liver and kidney."
                ),
            },
            "connectivity": {
                "type": "int",
                "default": 1,
                "range": [1, 3],
                "description": "Connectivity for FillHoles.",
            },
        },
    },
}


def get_tool_descriptions_for_llm() -> str:
    """Generate tool description text for the Strategist LLM."""
    lines = ["=== Available Toolbox ===\n"]
    for tool_id, info in TOOL_REGISTRY.items():
        lines.append(f"[{tool_id}] {info['name']} ({info['category']})")
        lines.append(f"  Description: {info['description']}")
        lines.append(f"  Use when: {info['applicable_when']}")
        if info["params"]:
            param_strs = []
            for pname, pinfo in info["params"].items():
                param_strs.append(
                    f"    {pname}: {pinfo['type']} (default={pinfo.get('default')})"
                )
            lines.append("  Parameters:")
            lines.extend(param_strs)
        lines.append("")
    return "\n".join(lines)


def get_filtered_tool_registry(allowed_categories: set) -> Dict[str, Dict]:
    """Return a filtered tool registry based on allowed categories.

    Args:
        allowed_categories: Set of category names to include (e.g., {"vfm_proposer", "postprocessing"})

    Returns:
        Filtered dictionary of tool_id -> tool_info
    """
    return {
        tool_id: info
        for tool_id, info in TOOL_REGISTRY.items()
        if info["category"] in allowed_categories
    }


def get_tool_descriptions_for_llm_filtered(allowed_categories: set) -> str:
    """Generate tool description text filtered by category.

    Args:
        allowed_categories: Set of category names to include

    Returns:
        Filtered tool descriptions for the LLM
    """
    filtered_registry = get_filtered_tool_registry(allowed_categories)
    lines = ["=== Available Toolbox ===\n"]
    for tool_id, info in filtered_registry.items():
        lines.append(f"[{tool_id}] {info['name']} ({info['category']})")
        lines.append(f"  Description: {info['description']}")
        lines.append(f"  Use when: {info['applicable_when']}")
        if info["params"]:
            param_strs = []
            for pname, pinfo in info["params"].items():
                param_strs.append(
                    f"    {pname}: {pinfo['type']} (default={pinfo.get('default')})"
                )
            lines.append("  Parameters:")
            lines.extend(param_strs)
        lines.append("")
    return "\n".join(lines)


# =============================================================================
#  NIfTI I/O helpers (thin wrappers around nibabel)
# =============================================================================


def _load_nii(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Any]:
    """Load a NIfTI volume -> (data, affine, header)."""
    obj = nib.load(path)
    return obj.get_fdata(), obj.affine, obj.header


def _save_nii(data: np.ndarray, path: str, affine=None, header=None):
    """Save an array as NIfTI."""
    if affine is None:
        affine = np.eye(4)
    dtype = np.uint8 if data.max() < 256 else np.int16
    img = nib.Nifti1Image(data.astype(dtype), affine, header)
    nib.save(img, path)


def _infer_axial_axis(volume_shape: tuple) -> int:
    """Infer the axial (slice) axis from a 3D medical image shape.

    Heuristic: the axial axis typically has the fewest slices.
    For ambiguous cases (e.g., isotropic), default to axis 2 (NIfTI RAS+ convention).
    """
    if len(volume_shape) != 3:
        return 2
    min_dim = min(volume_shape)
    min_axes = [i for i in range(3) if volume_shape[i] == min_dim]
    if len(min_axes) == 1:
        return min_axes[0]
    if 2 in min_axes:
        return 2
    return min_axes[0]


# =============================================================================
#  MONAI Post-processing tools
# =============================================================================


class PostprocessSegmentationTool:
    """
    MONAI Compose pipeline: RemoveSmallObjects → KeepLargestConnectedComponent → FillHoles.

    Operates per-label on multi-label segmentation masks.
    Replaces the old TopologyCleaner + HoleFilling tools.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("PostprocessSeg")

    def run(
        self, mask_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        min_size = params.get("min_size", 64)
        keep_largest = params.get("keep_largest_only", True)
        do_fill = params.get("fill_holes", True)
        connectivity = params.get("connectivity", 2)

        data, affine, header = _load_nii(mask_path)
        data = np.round(data).astype(np.int64)
        labels = [int(l) for l in np.unique(data) if l != 0]

        if not labels:
            _save_nii(data, output_path, affine, header)
            return output_path

        result = np.zeros_like(data, dtype=np.int64)

        for label_id in labels:
            binary = (data == label_id).astype(np.int64)
            # (C, H, W, D) — MONAI expects channel-first
            tensor = torch.from_numpy(binary[np.newaxis])

            # Build per-label pipeline
            steps = []
            if min_size > 0:
                steps.append(
                    RemoveSmallObjects(min_size=min_size, connectivity=connectivity)
                )
            if keep_largest:
                steps.append(
                    KeepLargestConnectedComponent(
                        applied_labels=[1], connectivity=connectivity
                    )
                )
            if do_fill:
                steps.append(FillHoles(applied_labels=[1], connectivity=connectivity))

            if steps:
                pipeline = Compose(steps)
                tensor = pipeline(tensor)

            cleaned = tensor.numpy().squeeze(0).astype(np.int64)
            result[cleaned > 0] = label_id

        _save_nii(result, output_path, affine, header)
        self.logger.info(
            f"PostprocessSegmentation completed → {os.path.basename(output_path)}"
        )
        return output_path


class GaussianSmoothMaskTool:
    """
    Smooth mask boundaries using MONAI GaussianSmooth.

    Per-label: cast to float → GaussianSmooth(sigma) → threshold at 0.5.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("GaussianSmoothMask")

    def run(
        self, mask_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        sigma = params.get("sigma", 1.0)

        data, affine, header = _load_nii(mask_path)
        data = np.round(data).astype(np.int64)
        labels = [int(l) for l in np.unique(data) if l != 0]
        result = np.zeros_like(data, dtype=np.int64)

        smoother = GaussianSmooth(sigma=sigma)

        for label_id in labels:
            binary = (data == label_id).astype(np.float32)
            # MONAI expects (C, H, W, D)
            tensor = torch.from_numpy(binary[np.newaxis])
            smoothed = smoother(tensor)
            result[smoothed.numpy().squeeze(0) > 0.5] = label_id

        _save_nii(result, output_path, affine, header)
        self.logger.info(
            f"GaussianSmoothMask completed → {os.path.basename(output_path)}"
        )
        return output_path


# =============================================================================
#  Morphological tools (MONAI dilate/erode — replaces scipy.ndimage)
# =============================================================================


class MorphologicalClosingTool:
    """Morphological closing (dilation → erosion) via MONAI dilate/erode.

    Per-label: binary mask → dilate → erode → reassemble multi-label.
    Requires (N, C, H, W, D) batched input for MONAI dilate/erode functions.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("MorphClosing")

    def run(
        self, mask_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        filter_size = params.get("filter_size", 3)
        iterations = params.get("iterations", 1)

        data, affine, header = _load_nii(mask_path)
        data = np.round(data).astype(np.int16)
        result = np.zeros_like(data)

        for label_id in [l for l in np.unique(data) if l != 0]:
            binary = (data == label_id).astype(np.float32)
            # MONAI dilate/erode expect (N, C, H, W, D)
            tensor = torch.from_numpy(binary[None, None])
            for _ in range(iterations):
                tensor = dilate(tensor, filter_size=filter_size)
                tensor = erode(tensor, filter_size=filter_size)
            result[tensor.squeeze().numpy() > 0.5] = label_id

        _save_nii(result, output_path, affine, header)
        self.logger.info(f"MorphClosing completed → {os.path.basename(output_path)}")
        return output_path


class MorphologicalOpeningTool:
    """Morphological opening (erosion → dilation) via MONAI dilate/erode.

    Per-label: binary mask → erode → dilate → reassemble multi-label.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("MorphOpening")

    def run(
        self, mask_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        filter_size = params.get("filter_size", 3)
        iterations = params.get("iterations", 1)

        data, affine, header = _load_nii(mask_path)
        data = np.round(data).astype(np.int16)
        result = np.zeros_like(data)

        for label_id in [l for l in np.unique(data) if l != 0]:
            binary = (data == label_id).astype(np.float32)
            # MONAI dilate/erode expect (N, C, H, W, D)
            tensor = torch.from_numpy(binary[None, None])
            for _ in range(iterations):
                tensor = erode(tensor, filter_size=filter_size)
                tensor = dilate(tensor, filter_size=filter_size)
            result[tensor.squeeze().numpy() > 0.5] = label_id

        _save_nii(result, output_path, affine, header)
        self.logger.info(f"MorphOpening completed → {os.path.basename(output_path)}")
        return output_path


# =============================================================================
#  MONAI Preprocessing tools (operate on image, not mask)
# =============================================================================


class EnhanceImageTool:
    """
    MONAI intensity enhancement for preprocessed [0, 255] images.

    Two modes (driven by perceptor image_profile features):
      - 'gamma':     AdjustContrast(gamma) — corrects intensity skew.
                     Use when intensity_skewness > 1.5 (right-skewed).
                     gamma < 1.0 brightens dark regions.
      - 'histogram': HistogramNormalize(num_bins) — equalizes multi-peak distributions.
                     Use when histogram_num_peaks > 5.

    Operates on the image (not mask). Saves enhanced image as NIfTI.
    Output range may differ from [0, 255] depending on mode.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("EnhanceImage")

    def run(
        self, image_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        mode = params.get("mode", "gamma")
        gamma = params.get("gamma", 0.7)
        num_bins = params.get("num_bins", 256)

        data, affine, header = _load_nii(image_path)

        # Convert to (C, H, W, D) float tensor for MONAI transforms
        if data.ndim == 3 and data.shape[2] == 3:
            # RGB 2D Image: convert (H, W, 3) to (3, H, W) -> C=3, spatial=(H, W)
            tensor = torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        elif data.ndim == 2:
            # Grayscale 2D Image: (H, W) to (1, H, W) -> C=1, spatial=(H, W)
            tensor = torch.from_numpy(data.astype(np.float32)[np.newaxis])
        else:
            # 3D NIfTI Image: (H, W, D) to (1, H, W, D) -> C=1, spatial=(H, W, D)
            tensor = torch.from_numpy(data.astype(np.float32)[np.newaxis])

        if mode == "gamma":
            transform = AdjustContrast(gamma=gamma)
            tensor = transform(tensor)
        elif mode == "histogram":
            transform = HistogramNormalize(num_bins=num_bins)
            tensor = transform(tensor)
        else:
            self.logger.warning(f"Unknown enhance mode '{mode}', defaulting to 'gamma'")
            transform = AdjustContrast(gamma=gamma)
            tensor = transform(tensor)

        # Back to numpy, squeeze channel dim
        if data.ndim == 3 and data.shape[2] == 3:
            result = tensor.numpy().transpose(1, 2, 0)
        else:
            result = tensor.numpy().squeeze(0)

        _save_nii(result, output_path, affine, header)
        self.logger.info(
            f"EnhanceImage ({mode}) completed → {os.path.basename(output_path)}"
        )
        return output_path


class SmoothImageTool:
    """
    MONAI noise reduction for images before VFM inference.

    Two methods (driven by perceptor image_profile features):
      - 'gaussian': GaussianSmooth(sigma) — general denoising.
                    Use when noise_level_auto == 'High' or snr_estimate < 15.
                    sigma ~= noise_std / 10 for calibrated denoising.
      - 'median':   MedianSmooth(radius) — edge-preserving denoising.
                    Better near organ boundaries.

    Operates on the image (not mask). Saves smoothed image as NIfTI.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("SmoothImage")

    def run(
        self, image_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        method = params.get("method", "gaussian")
        sigma = params.get("sigma", 1.0)
        radius = params.get("radius", 1)

        data, affine, header = _load_nii(image_path)

        # Convert to (C, H, W, D) float tensor for MONAI transforms
        if data.ndim == 3 and data.shape[2] == 3:
            # RGB 2D Image: convert (H, W, 3) to (3, H, W) -> C=3, spatial=(H, W)
            tensor = torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        elif data.ndim == 2:
            # Grayscale 2D Image: (H, W) to (1, H, W) -> C=1, spatial=(H, W)
            tensor = torch.from_numpy(data.astype(np.float32)[np.newaxis])
        else:
            # 3D NIfTI Image: (H, W, D) to (1, H, W, D) -> C=1, spatial=(H, W, D)
            tensor = torch.from_numpy(data.astype(np.float32)[np.newaxis])

        if method == "gaussian":
            transform = GaussianSmooth(sigma=sigma)
            tensor = transform(tensor)
        elif method == "median":
            transform = MedianSmooth(radius=radius)
            tensor = transform(tensor)
        else:
            self.logger.warning(f"Unknown smooth method '{method}', using 'gaussian'")
            transform = GaussianSmooth(sigma=sigma)
            tensor = transform(tensor)

        # Back to numpy, squeeze channel dim
        if data.ndim == 3 and data.shape[2] == 3:
            result = tensor.numpy().transpose(1, 2, 0)
        else:
            result = tensor.numpy().squeeze(0)

        _save_nii(result, output_path, affine, header)
        self.logger.info(
            f"SmoothImage ({method}) completed → {os.path.basename(output_path)}"
        )
        return output_path


# =============================================================================
#  MONAI Label Filter tool
# =============================================================================


class LabelFilterTool:
    """
    MONAI LabelFilter — removes specific labels from a multi-label mask.

    Keeps only the labels in applied_labels, sets all others to 0.
    Useful when the Evaluator identifies certain organs as poorly segmented,
    allowing them to be re-generated by VFM in a subsequent step.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("LabelFilter")

    def run(
        self, mask_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        applied_labels = params.get("applied_labels", [])

        if not applied_labels:
            self.logger.warning("No applied_labels specified, copying mask unchanged")
            shutil.copy(mask_path, output_path)
            return output_path

        data, affine, header = _load_nii(mask_path)
        data = np.round(data).astype(np.int64)

        # MONAI LabelFilter expects (C, H, W, D) tensor
        tensor = torch.from_numpy(data[np.newaxis].astype(np.float32))
        lf = LabelFilter(applied_labels=applied_labels)
        filtered = lf(tensor)

        result = filtered.numpy().squeeze(0).astype(np.int64)
        _save_nii(result, output_path, affine, header)
        self.logger.info(
            f"LabelFilter (keep={applied_labels}) completed → {os.path.basename(output_path)}"
        )
        return output_path


# =============================================================================
#  Diagnostic: Symmetry Checker
# =============================================================================


class SymmetryCheckerTool:
    """
    Check volume ratio and flip-overlap Dice for paired left/right organs.
    Returns diagnostic report (does NOT modify mask).
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("SymmetryChecker")

    def run(self, mask_path: str, params: Optional[Dict] = None) -> Dict:
        params = params or {}
        organ_pairs = params.get("organ_pairs", [])
        vol_thresh = params.get("vol_threshold", 0.2)
        dice_thresh = params.get("dice_threshold", 0.1)

        data, _, _ = _load_nii(mask_path)
        data = np.round(data).astype(np.int16)
        z_axis = _infer_axial_axis(data.shape)
        remaining = [i for i in range(3) if i != z_axis]
        flip_axis = remaining[0]

        report = {"is_abnormal": False, "abnormal_pairs": [], "details": {}}

        for pair in organ_pairs:
            keys = list(pair.keys())
            name_l, id_l = keys[0], pair[keys[0]]
            name_r, id_r = keys[1], pair[keys[1]]
            pair_name = f"{name_l}_vs_{name_r}"

            mask_l = data == id_l
            mask_r = data == id_r
            vol_l, vol_r = int(mask_l.sum()), int(mask_r.sum())

            if vol_l == 0 or vol_r == 0:
                continue

            # Flip + centroid align + Dice
            mask_r_f = np.flip(mask_r, axis=flip_axis)
            com_l = np.array(ndimage.center_of_mass(mask_l))
            com_r_f = np.array(ndimage.center_of_mass(mask_r_f))
            shift = com_l - com_r_f
            mask_r_aligned = ndimage.shift(mask_r_f.astype(np.float64), shift, order=0)

            inter = int(np.logical_and(mask_l, mask_r_aligned > 0.5).sum())
            union_sum = vol_l + int((mask_r_aligned > 0.5).sum())
            dice = float(2.0 * inter / union_sum) if union_sum > 0 else 0.0
            vol_ratio = float(min(vol_l, vol_r) / max(vol_l, vol_r))

            is_abn = (vol_ratio < vol_thresh) or (dice < dice_thresh)
            report["details"][pair_name] = {
                "volume_ratio": vol_ratio,
                "symmetry_dice": dice,
                "abnormal": is_abn,
            }
            if is_abn:
                report["is_abnormal"] = True
                report["abnormal_pairs"].append(pair_name)

        return report


# =============================================================================
#  New MONAI tools: NormalizeIntensity, ClipIntensity, IterativeHoleFill,
#  BoundaryRefinement, MultiLabelClosing, RemoveOutlierLabels, ConditionalFill
# =============================================================================


class NormalizeIntensityTool:
    """
    Window/level intensity normalization using MONAI ScaleIntensityRange.

    Clips input intensities to [a_min, a_max] and rescales to [b_min, b_max].
    Useful for standardizing CT HU ranges before VFM inference.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("NormalizeIntensity")

    def run(
        self, image_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        a_min = params.get("a_min", -1000.0)
        a_max = params.get("a_max", 1000.0)
        b_min = params.get("b_min", 0.0)
        b_max = params.get("b_max", 255.0)
        clip = params.get("clip", True)

        data, affine, header = _load_nii(image_path)
        if data.ndim == 3 and data.shape[2] == 3:
            # RGB 2D Image: convert (H, W, 3) to (3, H, W) -> C=3, spatial=(H, W)
            tensor = torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        elif data.ndim == 2:
            # Grayscale 2D Image: (H, W) to (1, H, W) -> C=1, spatial=(H, W)
            tensor = torch.from_numpy(data.astype(np.float32)[np.newaxis])
        else:
            # 3D NIfTI Image: (H, W, D) to (1, H, W, D) -> C=1, spatial=(H, W, D)
            tensor = torch.from_numpy(data.astype(np.float32)[np.newaxis])

        transform = ScaleIntensityRange(
            a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=clip
        )
        tensor = transform(tensor)

        if data.ndim == 3 and data.shape[2] == 3:
            result = tensor.numpy().transpose(1, 2, 0)
        else:
            result = tensor.numpy().squeeze(0)
        _save_nii(result, output_path, affine, header)
        self.logger.info(
            f"NormalizeIntensity [{a_min},{a_max}]→[{b_min},{b_max}] completed "
            f"→ {os.path.basename(output_path)}"
        )
        return output_path


class ClipIntensityTool:
    """
    Robust percentile-based intensity clipping using MONAI
    ScaleIntensityRangePercentiles.  More robust than fixed-window normalization.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("ClipIntensity")

    def run(
        self, image_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        lower = params.get("lower", 1.0)
        upper = params.get("upper", 99.0)
        b_min = params.get("b_min", 0.0)
        b_max = params.get("b_max", 255.0)

        data, affine, header = _load_nii(image_path)
        if data.ndim == 3 and data.shape[2] == 3:
            # RGB 2D Image: convert (H, W, 3) to (3, H, W) -> C=3, spatial=(H, W)
            tensor = torch.from_numpy(data.transpose(2, 0, 1).astype(np.float32))
        elif data.ndim == 2:
            # Grayscale 2D Image: (H, W) to (1, H, W) -> C=1, spatial=(H, W)
            tensor = torch.from_numpy(data.astype(np.float32)[np.newaxis])
        else:
            # 3D NIfTI Image: (H, W, D) to (1, H, W, D) -> C=1, spatial=(H, W, D)
            tensor = torch.from_numpy(data.astype(np.float32)[np.newaxis])

        transform = ScaleIntensityRangePercentiles(
            lower=lower, upper=upper, b_min=b_min, b_max=b_max, clip=True
        )
        tensor = transform(tensor)

        if data.ndim == 3 and data.shape[2] == 3:
            result = tensor.numpy().transpose(1, 2, 0)
        else:
            result = tensor.numpy().squeeze(0)
        _save_nii(result, output_path, affine, header)
        self.logger.info(
            f"ClipIntensity [{lower}%,{upper}%] completed "
            f"→ {os.path.basename(output_path)}"
        )
        return output_path


class IterativeHoleFillTool:
    """
    Repeatedly applies MONAI FillHoles with increasing connectivity to close
    progressively larger holes in mask regions.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("IterativeHoleFill")

    def run(
        self, mask_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        iterations = params.get("iterations", 3)
        connectivity = params.get("connectivity", 1)

        data, affine, header = _load_nii(mask_path)
        data = np.round(data).astype(np.int64)
        labels = [int(l) for l in np.unique(data) if l != 0]

        if not labels:
            _save_nii(data, output_path, affine, header)
            return output_path

        result = np.zeros_like(data, dtype=np.int64)

        for label_id in labels:
            binary = (data == label_id).astype(np.int64)
            tensor = torch.from_numpy(binary[np.newaxis])
            # Apply FillHoles multiple times for progressively larger holes
            for _ in range(iterations):
                fill = FillHoles(applied_labels=[1], connectivity=connectivity)
                tensor = fill(tensor)
            cleaned = tensor.numpy().squeeze(0).astype(np.int64)
            result[cleaned > 0] = label_id

        _save_nii(result, output_path, affine, header)
        self.logger.info(
            f"IterativeHoleFill (iterations={iterations}) completed "
            f"→ {os.path.basename(output_path)}"
        )
        return output_path


class BoundaryRefinementTool:
    """
    Sharpens mask boundaries via erosion followed by selective dilation.
    Removes thin fuzzy boundary halos then re-expands to recover organ extent.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("BoundaryRefinement")

    def run(
        self, mask_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        erode_size = params.get("erode_size", 3)
        dilate_size = params.get("dilate_size", 3)
        iterations = params.get("iterations", 1)

        data, affine, header = _load_nii(mask_path)
        data = np.round(data).astype(np.int16)
        result = np.zeros_like(data)

        for label_id in [l for l in np.unique(data) if l != 0]:
            binary = (data == label_id).astype(np.float32)
            tensor = torch.from_numpy(binary[None, None])
            for _ in range(iterations):
                tensor = erode(tensor, filter_size=erode_size)
                tensor = dilate(tensor, filter_size=dilate_size)
            result[tensor.squeeze().numpy() > 0.5] = label_id

        _save_nii(result, output_path, affine, header)
        self.logger.info(
            f"BoundaryRefinement (erode={erode_size}, dilate={dilate_size}) completed "
            f"→ {os.path.basename(output_path)}"
        )
        return output_path


class MultiLabelClosingTool:
    """
    Per-label morphological closing with label-specific kernel sizes.
    Allows different organs to receive different closing strengths.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("MultiLabelClosing")

    def run(
        self, mask_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        default_filter_size = params.get("default_filter_size", 3)
        label_sizes: Dict = params.get("label_sizes", {})
        iterations = params.get("iterations", 1)

        # Normalize label_sizes keys to int
        label_sizes = {int(k): int(v) for k, v in label_sizes.items()}

        data, affine, header = _load_nii(mask_path)
        data = np.round(data).astype(np.int16)
        result = np.zeros_like(data)

        for label_id in [l for l in np.unique(data) if l != 0]:
            filter_size = label_sizes.get(label_id, default_filter_size)
            binary = (data == label_id).astype(np.float32)
            tensor = torch.from_numpy(binary[None, None])
            for _ in range(iterations):
                tensor = dilate(tensor, filter_size=filter_size)
                tensor = erode(tensor, filter_size=filter_size)
            result[tensor.squeeze().numpy() > 0.5] = label_id

        _save_nii(result, output_path, affine, header)
        self.logger.info(
            f"MultiLabelClosing completed → {os.path.basename(output_path)}"
        )
        return output_path


class RemoveOutlierLabelsTool:
    """
    Removes segmentation fragments spatially far from the main organ body.
    Uses centroid distance filtering per label.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("RemoveOutlierLabels")

    def run(
        self, mask_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        from scipy import ndimage as ndi

        params = params or {}
        max_distance = params.get("max_distance", 50.0)
        min_size = params.get("min_size", 10)

        data, affine, header = _load_nii(mask_path)
        data = np.round(data).astype(np.int64)
        result = np.zeros_like(data, dtype=np.int64)

        for label_id in [int(l) for l in np.unique(data) if l != 0]:
            binary = (data == label_id).astype(np.int32)
            labeled, num_features = ndi.label(binary)

            if num_features == 0:
                continue
            if num_features == 1:
                result[binary > 0] = label_id
                continue

            # Find the largest component centroid
            comp_sizes = [(labeled == c).sum() for c in range(1, num_features + 1)]
            largest_comp = int(np.argmax(comp_sizes)) + 1
            largest_mask = labeled == largest_comp
            main_centroid = np.array(ndi.center_of_mass(largest_mask))

            for comp_id in range(1, num_features + 1):
                comp_mask = labeled == comp_id
                comp_size = comp_sizes[comp_id - 1]
                if comp_size < min_size:
                    continue
                centroid = np.array(ndi.center_of_mass(comp_mask))
                dist = float(np.linalg.norm(centroid - main_centroid))
                if comp_id == largest_comp or dist <= max_distance:
                    result[comp_mask] = label_id

        _save_nii(result, output_path, affine, header)
        self.logger.info(
            f"RemoveOutlierLabels (max_dist={max_distance}) completed "
            f"→ {os.path.basename(output_path)}"
        )
        return output_path


class ConditionalFillTool:
    """
    Fills holes in specific labels only when their volume is below expected minimum.
    Prevents over-aggressive filling on well-segmented organs.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("ConditionalFill")

    def run(
        self, mask_path: str, output_path: str, params: Optional[Dict] = None
    ) -> str:
        params = params or {}
        expected_min_voxels: Dict = params.get("expected_min_voxels", {})
        connectivity = params.get("connectivity", 1)

        # Normalize keys to int
        expected_min_voxels = {int(k): int(v) for k, v in expected_min_voxels.items()}

        data, affine, header = _load_nii(mask_path)
        data = np.round(data).astype(np.int64)
        labels = [int(l) for l in np.unique(data) if l != 0]

        if not labels:
            _save_nii(data, output_path, affine, header)
            return output_path

        result = data.copy()

        for label_id in labels:
            min_voxels = expected_min_voxels.get(label_id, None)
            if min_voxels is None:
                continue  # Not specified — skip

            current_vol = int((data == label_id).sum())
            if current_vol < min_voxels:
                # Below expected volume — apply FillHoles to recover volume
                binary = (data == label_id).astype(np.int64)
                tensor = torch.from_numpy(binary[np.newaxis])
                fill = FillHoles(applied_labels=[1], connectivity=connectivity)
                tensor = fill(tensor)
                filled = tensor.numpy().squeeze(0).astype(np.int64)
                # Only add filled voxels to result (don't overwrite other labels)
                new_voxels = (filled > 0) & (result == 0)
                result[new_voxels] = label_id
                added = int(new_voxels.sum())
                self.logger.info(
                    f"ConditionalFill: label {label_id} vol={current_vol} < "
                    f"min={min_voxels}, added {added} voxels"
                )

        _save_nii(result, output_path, affine, header)
        self.logger.info(f"ConditionalFill completed → {os.path.basename(output_path)}")
        return output_path


# =============================================================================
#  nnUNet ↔ BiomedParse Data Converter
# =============================================================================


class NNUNet2BiomedParseConverter:
    """
    Convert nnUNet dataset (imagesTr/ + labelsTr/) to BiomedParse NPZ format.

    NPZ structure per case:
      imgs:  uint8 array (W, H, D) — intensity normalized to [0, 255]
      gts:   uint8 array (W, H, D) — label mask
      text_prompts: dict {"1": "organ description", "instance_label": 0}
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("NNUNet2BP")

    def convert(
        self,
        input_dir: str,
        output_dir: str,
        text_prompts: Dict[str, Any],
    ) -> int:
        """
        Convert all cases in nnUNet dir to BiomedParse NPZ.

        Args:
            input_dir: nnUNet dataset root with imagesTr/ and labelsTr/
            output_dir: output directory for .npz files
            text_prompts: e.g. {"1": "Liver in Abdomen CT", "instance_label": 0}

        Returns:
            Number of successfully converted cases.
        """
        images_dir = Path(input_dir) / "imagesTr"
        labels_dir = Path(input_dir) / "labelsTr"
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if not images_dir.exists():
            raise ValueError(f"imagesTr not found: {images_dir}")
        if not labels_dir.exists():
            raise ValueError(f"labelsTr not found: {labels_dir}")

        # Find image files (*_0000.nii.gz or *_0000.png)
        image_files = sorted(images_dir.glob("*_0000.nii.gz"))
        image_files += sorted(images_dir.glob("*_0000.png"))

        if not image_files:
            self.logger.warning(f"No image files found in {images_dir}")
            return 0

        self.logger.info(
            f"Converting {len(image_files)} cases from {input_dir} → {output_dir}"
        )

        count = 0
        for img_file in image_files:
            try:
                ok = self._convert_one(img_file, labels_dir, out_path, text_prompts)
                if ok:
                    count += 1
            except Exception as e:
                self.logger.error(f"Failed to convert {img_file.name}: {e}")

        self.logger.info(f"Converted {count}/{len(image_files)} cases.")
        return count

    def _convert_one(
        self,
        img_file: Path,
        labels_dir: Path,
        out_path: Path,
        text_prompts: Dict,
    ) -> bool:
        """Convert a single case."""
        # Determine base name
        if img_file.name.endswith("_0000.nii.gz"):
            base_name = img_file.name.replace("_0000.nii.gz", "")
        elif img_file.name.endswith("_0000.png"):
            base_name = img_file.name.replace("_0000.png", "")
        else:
            return False

        # Find matching label
        label_file = None
        for suffix in [".nii.gz", ".png"]:
            candidate = labels_dir / f"{base_name}{suffix}"
            if candidate.exists():
                label_file = candidate
                break

        if label_file is None:
            self.logger.warning(f"No label found for {base_name}")
            return False

        # Load
        img_data = self._load_any(img_file)
        lbl_data = self._load_any(label_file)
        if img_data is None or lbl_data is None:
            return False

        if img_data.shape != lbl_data.shape:
            self.logger.warning(
                f"Shape mismatch for {base_name}: "
                f"img={img_data.shape} lbl={lbl_data.shape}"
            )
            return False

        # Normalize image to uint8
        img_data = self._normalize_to_uint8(img_data)
        lbl_data = lbl_data.astype(np.uint8)

        # Save NPZ
        npz_file = out_path / f"{base_name}.npz"
        np.savez_compressed(
            npz_file,
            imgs=img_data,
            gts=lbl_data,
            text_prompts=json.dumps(text_prompts),
        )
        return True

    @staticmethod
    def _load_any(filepath: Path) -> Optional[np.ndarray]:
        """Load NIfTI or PNG."""
        name = filepath.name
        if name.endswith(".nii.gz") or name.endswith(".nii"):
            obj = nib.load(str(filepath))
            return obj.get_fdata()
        elif name.endswith(".png"):
            from PIL import Image

            arr = np.array(Image.open(str(filepath)))
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            return arr
        return None

    @staticmethod
    def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
        """Normalize intensity to [0, 255] uint8 using 1-99 percentile clipping."""
        p_low, p_high = np.percentile(data, [1, 99])
        data = np.clip(data, p_low, p_high).astype(np.float64)
        denom = data.max() - data.min()
        if denom > 1e-6:
            data = (data - data.min()) / denom * 255.0
        else:
            data = np.zeros_like(data)
        return data.astype(np.uint8)

    @staticmethod
    def verify_npz(npz_path: str) -> Dict[str, Any]:
        """Verify a converted NPZ file and return summary info."""
        data = np.load(npz_path, allow_pickle=True)
        info = {"path": npz_path, "keys": list(data.keys())}
        if "imgs" in data:
            info["imgs_shape"] = data["imgs"].shape
            info["imgs_dtype"] = str(data["imgs"].dtype)
            info["imgs_range"] = [int(data["imgs"].min()), int(data["imgs"].max())]
        if "gts" in data:
            info["gts_shape"] = data["gts"].shape
            info["gts_unique"] = sorted(int(v) for v in np.unique(data["gts"]))
        if "text_prompts" in data:
            info["text_prompts"] = data["text_prompts"].item()
        data.close()
        return info


# =============================================================================
#  VFM Wrappers (SAM3 subprocess, BiomedParse multiprocessing)
# =============================================================================


class SAM3Tool:
    """SAM3 inference via subprocess to separate conda env.

    Supports dual-GPU execution (preferred for speed) with automatic fallback
    to single-GPU when NCCL communication errors or OOM are detected.
    Once a fallback GPU is chosen for this session, subsequent calls skip
    the dual-GPU attempt to avoid repeated failures.
    """

    # Error patterns that indicate multi-GPU issues warranting single-GPU fallback
    _MULTI_GPU_ERROR_PATTERNS = (
        "nccl",
        "OutOfMemoryError",
        "out of memory",
        "CUDA error",
        "Segmentation fault",
        "SIGABRT",
        "Aborted",
    )

    def __init__(
        self,
        python_path: str,
        script_path: str,
        checkpoint: str,
        gpus: List[int],
        sam3_root: str = "",
        logger: Optional[logging.Logger] = None,
    ):
        self.python_path = python_path
        self.script_path = os.path.abspath(script_path)
        self.checkpoint = checkpoint
        self.gpus = gpus
        self.sam3_root = sam3_root
        self.logger = logger or logging.getLogger("SAM3Tool")
        # Once set, this overrides self.gpus for all future calls in this session
        self._fallback_gpu: Optional[List[int]] = None

    def _run_once(
        self, config_json_path: str, output_dir: str, gpus: List[int]
    ) -> None:
        """Run SAM3 subprocess with the given GPU list. Raises on failure."""
        cmd = [
            self.python_path,
            self.script_path,
            "--json_config",
            config_json_path,
            "--output_dir",
            output_dir,
            "--checkpoint",
            self.checkpoint,
            "--gpus",
            ",".join(map(str, gpus)),
        ]
        if self.sam3_root:
            cmd += ["--sam3_root", self.sam3_root]
        self.logger.info(f"Running SAM3 (gpus={gpus}): {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3600,  # 60 min — multi-view 512² inference exceeds 30 min on real volumes
            )
            self.logger.info(f"SAM3 stdout:\n{result.stdout[-2000:]}")
        except subprocess.TimeoutExpired:
            self.logger.error(f"SAM3 timed out after 1800s (gpus={gpus})")
            raise
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SAM3 failed (gpus={gpus}):\n{e.stderr[-2000:]}")
            raise

    @staticmethod
    def _is_multi_gpu_error(stderr: str) -> bool:
        """Return True if the error looks like a multi-GPU (NCCL/OOM) issue."""
        lower = stderr.lower()
        return any(pat.lower() in lower for pat in SAM3Tool._MULTI_GPU_ERROR_PATTERNS)

    def run(self, config_json_path: str, output_dir: str) -> str:
        """Run SAM3 inference, with automatic single-GPU fallback on NCCL/OOM errors.

        Fallback order when dual-GPU fails:
          1. Try gpus[-1] alone  (e.g. GPU 1 — less likely to be saturated)
          2. Try gpus[0] alone   (e.g. GPU 0)
        Once a fallback GPU succeeds, it is remembered for the lifetime of this
        SAM3Tool instance so subsequent calls skip the dual-GPU attempt.
        """
        os.makedirs(output_dir, exist_ok=True)

        # If a previous call already established a stable fallback, use it directly
        if self._fallback_gpu is not None:
            self._run_once(config_json_path, output_dir, self._fallback_gpu)
            return output_dir

        # First attempt: use configured GPU list (may be dual-GPU)
        if len(self.gpus) <= 1:
            # Already single-GPU — no fallback logic needed
            self._run_once(config_json_path, output_dir, self.gpus)
            return output_dir

        try:
            self._run_once(config_json_path, output_dir, self.gpus)
            return output_dir
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            if not self._is_multi_gpu_error(stderr):
                # Not a multi-GPU issue — re-raise immediately
                raise

            self.logger.warning(
                "SAM3 dual-GPU attempt failed with multi-GPU error "
                "(NCCL/OOM). Will retry with single-GPU fallback."
            )

        # Build single-GPU candidates: prefer last GPU, then first GPU
        single_gpu_candidates = []
        if len(self.gpus) >= 2:
            single_gpu_candidates.append([self.gpus[-1]])  # e.g. [1]
        single_gpu_candidates.append([self.gpus[0]])  # e.g. [0]

        last_exc: Optional[Exception] = None
        for fallback in single_gpu_candidates:
            try:
                self.logger.info(f"SAM3 single-GPU fallback attempt: gpus={fallback}")
                self._run_once(config_json_path, output_dir, fallback)
                # Success — remember this GPU for the rest of the session
                self._fallback_gpu = fallback
                self.logger.info(
                    f"SAM3 single-GPU fallback succeeded (gpus={fallback}). "
                    f"This GPU will be used for all subsequent SAM3 calls."
                )
                return output_dir
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                self.logger.warning(
                    f"SAM3 single-GPU fallback also failed (gpus={fallback}): {exc}"
                )
                last_exc = exc

        # All fallback options exhausted
        raise RuntimeError(
            f"SAM3 failed on all GPU configurations tried: "
            f"{[self.gpus] + single_gpu_candidates}"
        ) from last_exc


class BiomedParseTool:
    """BiomedParse inference via subprocess to separate conda env (biomedparsev2).

    Analogous to SAM3Tool: calls run_biomedparse_standalone.py under
    the biomedparsev2 conda environment to avoid dependency conflicts.

    Output structure:
        output_dir/
            {case}.npz          — NPZ predictions (always)
            nii_predictions/
                {case}_pred.nii.gz  — NIfTI predictions (if save_nii=True)
    """

    def __init__(
        self,
        python_path: str,
        script_path: str,
        checkpoint: str,
        gpus: List[int],
        biomedparse_root: str = "",
        logger: Optional[logging.Logger] = None,
    ):
        self.python_path = python_path
        self.script_path = os.path.abspath(script_path)
        self.checkpoint = checkpoint
        self.gpus = gpus
        self.biomedparse_root = biomedparse_root
        self.logger = logger or logging.getLogger("BiomedParseTool")

    def run(
        self, data_path: str, output_dir: str, params: Optional[Dict] = None
    ) -> str:
        """Run BiomedParse inference via subprocess.

        Args:
            data_path: Directory containing .npz input files.
            output_dir: Directory for output predictions.
            params: Optional dict with keys:
                slice_batch_size (int), interpolate_size (int),
                nms_threshold (float), score_threshold (float),
                save_nii (bool).

        Returns:
            output_dir path.
        """
        os.makedirs(output_dir, exist_ok=True)
        params = params or {}

        cmd = [
            self.python_path,
            self.script_path,
            "--input_dir",
            data_path,
            "--output_dir",
            output_dir,
            "--checkpoint",
            self.checkpoint,
            "--gpus",
            ",".join(map(str, self.gpus)),
            "--slice_batch_size",
            str(params.get("slice_batch_size", 8)),
            "--interpolate_size",
            str(params.get("interpolate_size", 512)),
            "--nms_threshold",
            str(params.get("nms_threshold", 0.5)),
            "--score_threshold",
            str(params.get("score_threshold", 0.5)),
        ]
        if params.get("save_nii", True):
            cmd.append("--save_nii")
        if params.get("save_prob", False):
            cmd.append("--save_prob")
        if self.biomedparse_root:
            cmd += ["--biomedparse_root", self.biomedparse_root]

        self.logger.info(f"Running BiomedParse: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3600,  # 60 min — multi-view 512² inference exceeds 30 min on real volumes
            )
            self.logger.info(f"BiomedParse stdout:\n{result.stdout[-2000:]}")
            if result.stderr:
                self.logger.debug(f"BiomedParse stderr:\n{result.stderr[-2000:]}")
        except subprocess.TimeoutExpired:
            self.logger.error("BiomedParse timed out after 1800s")
            raise
        except subprocess.CalledProcessError as e:
            self.logger.error(f"BiomedParse failed:\n{e.stderr[-2000:]}")
            raise

        self.logger.info(f"BiomedParse completed → {output_dir}")
        return output_dir


# =============================================================================
#  ToolExecutor: unified dispatcher
# =============================================================================


class ToolExecutor:
    """
    Unified tool dispatcher — Executor Agent routes actions through this class.
    Maps tool_id + params to the corresponding tool implementation.
    """

    def __init__(self, config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("ToolExecutor")

        self.tools: Dict[str, Any] = {
            "postprocess_segmentation": PostprocessSegmentationTool(self.logger),
            "gaussian_smooth_mask": GaussianSmoothMaskTool(self.logger),
            "morphological_closing": MorphologicalClosingTool(self.logger),
            "morphological_opening": MorphologicalOpeningTool(self.logger),
            "label_filter": LabelFilterTool(self.logger),
            "enhance_image": EnhanceImageTool(self.logger),
            "smooth_image": SmoothImageTool(self.logger),
            "normalize_intensity": NormalizeIntensityTool(self.logger),
            "clip_intensity": ClipIntensityTool(self.logger),
            "iterative_hole_fill": IterativeHoleFillTool(self.logger),
            "boundary_refinement": BoundaryRefinementTool(self.logger),
            "multi_label_closing": MultiLabelClosingTool(self.logger),
            "remove_outlier_labels": RemoveOutlierLabelsTool(self.logger),
            "conditional_fill": ConditionalFillTool(self.logger),
            "symmetry_checker": SymmetryCheckerTool(self.logger),
        }

        self._bp_tool = None
        self._sam3_tool = None

    @property
    def biomedparse(self) -> "BiomedParseTool":
        if self._bp_tool is None:
            self._bp_tool = BiomedParseTool(
                python_path=self.config.paths.biomedparse_python,
                script_path=self.config.paths.biomedparse_script,
                checkpoint=self.config.paths.biomedparse_checkpoint,
                gpus=self.config.gpus,
                biomedparse_root=self.config.paths.biomedparse_root,
                logger=self.logger,
            )
        return self._bp_tool

    @property
    def sam3(self) -> "SAM3Tool":
        if self._sam3_tool is None:
            self._sam3_tool = SAM3Tool(
                python_path=self.config.paths.sam3_python,
                script_path=self.config.paths.sam3_script,
                checkpoint=self.config.paths.sam3_checkpoint,
                gpus=self.config.toolbox.sam3_gpus,
                sam3_root=self.config.paths.sam3_root,
                logger=self.logger,
            )
        return self._sam3_tool

    def execute(
        self,
        tool_id: str,
        image_path: str,
        mask_path: str,
        output_path: str,
        params: Optional[Dict] = None,
    ) -> str:
        """
        Execute a tool.

        Args:
            tool_id: key from TOOL_REGISTRY
            image_path: original image path
            mask_path: current intermediate mask path
            output_path: output mask/image path
            params: tool-specific parameters

        Returns:
            output_path
        """
        params = params or {}
        self.logger.info(f"Executing tool [{tool_id}] params={params}")

        # Preprocessing tools operate on image_path
        if tool_id in (
            "enhance_image",
            "smooth_image",
            "normalize_intensity",
            "clip_intensity",
        ):
            return self.tools[tool_id].run(image_path, output_path, params)

        # Postprocessing tools operate on mask_path
        elif tool_id in (
            "postprocess_segmentation",
            "gaussian_smooth_mask",
            "morphological_closing",
            "morphological_opening",
            "label_filter",
            "iterative_hole_fill",
            "boundary_refinement",
            "multi_label_closing",
            "remove_outlier_labels",
            "conditional_fill",
        ):
            return self.tools[tool_id].run(mask_path, output_path, params)

        # Diagnostics — don't modify mask, copy and return report
        elif tool_id == "symmetry_checker":
            report = self.tools["symmetry_checker"].run(mask_path, params)
            shutil.copy(mask_path, output_path)
            self.logger.info(f"Symmetry report: {json.dumps(report, indent=2)}")
            return output_path

        # VFM tools require special orchestration by Executor agent
        elif tool_id == "sam3":
            raise NotImplementedError(
                f"{tool_id} requires config generation by Executor agent. "
                f"Use ToolExecutor.sam3.run(config_path, output_dir) directly."
            )
        elif tool_id == "biomedparse":
            raise NotImplementedError(
                "BiomedParse requires NPZ data preparation. "
                "Use ToolExecutor.biomedparse.run(data_path, output_dir) directly."
            )
        elif tool_id == "dual_expert":
            raise NotImplementedError(
                "DualExpert requires BiomedParse → SAM3 orchestration. "
                "Use Executor agent's dual_expert pipeline directly."
            )
        else:
            raise ValueError(f"Unknown tool_id: {tool_id}")

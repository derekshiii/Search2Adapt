"""
Search2Adapt Ground-Truth Metrics Module

Computes per-organ Dice and ASD (Average Surface Distance) between
predicted pseudo-label masks and ground-truth NIfTI segmentations.

Key use-cases:
  1. Post-MCTS evaluation: validate how well LLM rewards correlate with
     real segmentation quality (reward credibility, Critique 1).
  2. Per-step quality tracking: trace whether each tool application
     actually improves the mask (ground-truth oracle signal).
  3. Batch aggregation: summarise results across multiple cases for
     ablation comparisons and paper tables.

The module is intentionally self-contained — it only depends on numpy,
scipy, and nibabel, all of which are available in the MONAI environment.
It has NO dependency on any Search2Adapt agent class so it can be
imported anywhere without circular imports.

ORGAN_LABEL_MAP (AMOS-CT Dataset003):
  0 = Background
  1 = Spleen
  2 = Right Kidney
  3 = Left Kidney
  4 = Gallbladder
  5 = Esophagus
  6 = Liver
  7 = Stomach
  8 = Aorta
  9 = Inferior Vena Cava
  10 = Pancreas
  11 = Right Adrenal Gland
  12 = Left Adrenal Gland
  13 = Duodenum
  14 = Bladder
  15 = Prostate or Uterus
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import nibabel as nib

    _HAS_NIBABEL = True
except ImportError:
    _HAS_NIBABEL = False

try:
    from scipy.ndimage import binary_erosion as _bin_erosion
    from scipy.ndimage import distance_transform_edt

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Default AMOS-CT organ label mapping (label_id → organ_name)
# ---------------------------------------------------------------------------

AMOS_LABEL_NAMES: Dict[int, str] = {
    1: "Spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Gallbladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "Inferior Vena Cava",
    10: "Pancreas",
    11: "Right Adrenal Gland",
    12: "Left Adrenal Gland",
    13: "Duodenum",
    14: "Bladder",
    15: "Prostate or Uterus",
}


# ---------------------------------------------------------------------------
#  Low-level metric primitives
# ---------------------------------------------------------------------------


def compute_dice(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """
    Compute Sørensen-Dice coefficient for a single binary label.

    Returns:
        Dice ∈ [0, 1], or float('nan') if both masks are empty
        (label absent in both pred AND gt — should be excluded from mean).
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = int(np.logical_and(pred, gt).sum())
    denominator = int(pred.sum()) + int(gt.sum())

    if denominator == 0:
        return float("nan")  # label absent in both → exclude from mean
    return (2.0 * intersection) / denominator


def compute_surface_distances(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Compute symmetric surface distances between pred and gt binary masks.

    Uses the border-voxel approach:
      border = mask & ~erode(mask)

    Returns:
        1-D numpy array of all surface distances in mm,
        or an empty array if either surface is degenerate / mask is empty.
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for surface distance computation")

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    if not np.any(pred) or not np.any(gt):
        return np.array([], dtype=np.float32)

    struct = np.ones((3,) * pred.ndim, dtype=bool)
    pred_border = pred & ~_bin_erosion(pred, structure=struct)
    gt_border = gt & ~_bin_erosion(gt, structure=struct)

    if not np.any(pred_border) or not np.any(gt_border):
        return np.array([], dtype=np.float32)

    # One-sided distances: pred surface → gt surface and vice versa
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    return np.concatenate([d_pred_to_gt, d_gt_to_pred]).astype(np.float32)


def compute_asd(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute Average Surface Distance (ASD) in mm.

    Returns:
        Mean surface distance ∈ [0, ∞), or float('nan') if either mask is
        empty (so the caller can decide whether to penalise or skip).
    """
    surf_dist = compute_surface_distances(pred_mask, gt_mask, spacing)
    if len(surf_dist) == 0:
        return float("nan")
    return float(np.mean(surf_dist))


def compute_hd95(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute 95th-percentile Hausdorff Distance (HD95) in mm.

    Returns float('nan') if either mask is empty.
    """
    surf_dist = compute_surface_distances(pred_mask, gt_mask, spacing)
    if len(surf_dist) == 0:
        return float("nan")
    return float(np.percentile(surf_dist, 95))


# ---------------------------------------------------------------------------
#  Per-case evaluation
# ---------------------------------------------------------------------------


def compute_case_metrics(
    pred_path: str,
    gt_path: str,
    label_ids: Optional[List[int]] = None,
    label_names: Optional[Dict[int, str]] = None,
    compute_hd: bool = False,
) -> Dict:
    """
    Compute per-organ Dice (+ASD, optionally HD95) for one case.

    Args:
        pred_path:   Path to predicted pseudo-label NIfTI (.nii.gz).
        gt_path:     Path to ground-truth NIfTI (.nii.gz).
        label_ids:   List of integer label IDs to evaluate (default: all
                     non-zero labels present in GT).
        label_names: Dict mapping label_id → organ_name for readable output.
                     Defaults to AMOS_LABEL_NAMES.
        compute_hd:  Whether to also compute HD95 (slower; default False).

    Returns:
        {
          "case_id":        str  (derived from pred_path filename),
          "pred_path":      str,
          "gt_path":        str,
          "per_organ": {
              "<organ_name>": {
                  "label_id":    int,
                  "dice":        float | nan,
                  "asd":         float | nan,  (mm)
                  "hd95":        float | nan,  (mm, if compute_hd)
                  "gt_voxels":   int,
                  "pred_voxels": int,
              },
              ...
          },
          "mean_dice":      float,   (nanmean over labels present in GT)
          "mean_asd":       float,   (nanmean over valid asd values)
          "mean_hd95":      float,   (nanmean if compute_hd)
          "num_valid_dice": int,     (labels with non-nan Dice)
          "spacing":        list[float],
        }
    """
    if not _HAS_NIBABEL:
        raise ImportError("nibabel is required for compute_case_metrics")

    if label_names is None:
        label_names = AMOS_LABEL_NAMES

    # Load volumes
    pred_obj = nib.load(pred_path)
    gt_obj = nib.load(gt_path)

    pred_data = np.round(pred_obj.get_fdata()).astype(np.int16)
    gt_data = np.round(gt_obj.get_fdata()).astype(np.int16)

    # Voxel spacing (mm) from NIfTI header — used for surface distances
    spacing = tuple(float(s) for s in gt_obj.header.get_zooms()[:3])
    if any(s <= 0 for s in spacing):
        spacing = (1.0, 1.0, 1.0)

    # Determine which labels to evaluate
    if label_ids is None:
        gt_unique = sorted(int(x) for x in np.unique(gt_data) if x > 0)
        label_ids = gt_unique

    per_organ: Dict = {}
    dice_vals: List[float] = []
    asd_vals: List[float] = []
    hd95_vals: List[float] = []

    for lid in label_ids:
        pred_mask = pred_data == lid
        gt_mask = gt_data == lid

        gt_vox = int(gt_mask.sum())
        pred_vox = int(pred_mask.sum())

        dice = compute_dice(pred_mask, gt_mask)
        asd = compute_asd(pred_mask, gt_mask, spacing)

        organ_entry: Dict = {
            "label_id": lid,
            "dice": dice,
            "asd": asd,
            "gt_voxels": gt_vox,
            "pred_voxels": pred_vox,
        }

        if compute_hd:
            hd95 = compute_hd95(pred_mask, gt_mask, spacing)
            organ_entry["hd95"] = hd95
            if not np.isnan(hd95):
                hd95_vals.append(hd95)

        name = label_names.get(lid, f"label_{lid}")
        per_organ[name] = organ_entry

        # Collect for mean computation: only labels present in GT
        if gt_vox > 0:
            if not np.isnan(dice):
                dice_vals.append(dice)
            if not np.isnan(asd):
                asd_vals.append(asd)

    # Case-level summary
    mean_dice = float(np.mean(dice_vals)) if dice_vals else float("nan")
    mean_asd = float(np.mean(asd_vals)) if asd_vals else float("nan")

    case_id = os.path.basename(pred_path).replace(".nii.gz", "").replace(".nii", "")

    result: Dict = {
        "case_id": case_id,
        "pred_path": pred_path,
        "gt_path": gt_path,
        "per_organ": per_organ,
        "mean_dice": mean_dice,
        "mean_asd": mean_asd,
        "num_valid_dice": len(dice_vals),
        "spacing": list(spacing),
    }

    if compute_hd:
        result["mean_hd95"] = float(np.mean(hd95_vals)) if hd95_vals else float("nan")

    return result


# ---------------------------------------------------------------------------
#  Batch evaluation
# ---------------------------------------------------------------------------


def compute_batch_metrics(
    case_metrics: List[Dict],
) -> Dict:
    """
    Aggregate a list of per-case metric dicts (from compute_case_metrics)
    into dataset-level statistics.

    Args:
        case_metrics: list of dicts returned by compute_case_metrics().

    Returns:
        {
          "num_cases":        int,
          "mean_dice":        float,   (macro-average across all cases)
          "std_dice":         float,
          "mean_asd":         float,
          "std_asd":          float,
          "per_organ_mean_dice": {organ_name: float, ...},
          "per_organ_mean_asd":  {organ_name: float, ...},
          "per_case_mean_dice":  {case_id: float, ...},
        }
    """
    if not case_metrics:
        return {"num_cases": 0, "mean_dice": float("nan"), "mean_asd": float("nan")}

    # Collect per-organ across all cases
    organ_dice: Dict[str, List[float]] = {}
    organ_asd: Dict[str, List[float]] = {}
    case_dice_list: List[float] = []

    for cm in case_metrics:
        if not np.isnan(cm["mean_dice"]):
            case_dice_list.append(cm["mean_dice"])
        for organ, info in cm.get("per_organ", {}).items():
            d = info.get("dice", float("nan"))
            a = info.get("asd", float("nan"))
            # only include if GT is present for this organ in this case
            if info.get("gt_voxels", 0) > 0:
                if not np.isnan(d):
                    organ_dice.setdefault(organ, []).append(d)
                if not np.isnan(a):
                    organ_asd.setdefault(organ, []).append(a)

    per_organ_mean_dice = {k: float(np.mean(v)) for k, v in organ_dice.items()}
    per_organ_mean_asd = {k: float(np.mean(v)) for k, v in organ_asd.items()}
    per_case_mean_dice = {cm["case_id"]: cm["mean_dice"] for cm in case_metrics}

    all_asds = [a for v in organ_asd.values() for a in v]

    return {
        "num_cases": len(case_metrics),
        "mean_dice": float(np.mean(case_dice_list)) if case_dice_list else float("nan"),
        "std_dice": float(np.std(case_dice_list)) if case_dice_list else float("nan"),
        "mean_asd": float(np.mean(all_asds)) if all_asds else float("nan"),
        "std_asd": float(np.std(all_asds)) if all_asds else float("nan"),
        "per_organ_mean_dice": per_organ_mean_dice,
        "per_organ_mean_asd": per_organ_mean_asd,
        "per_case_mean_dice": per_case_mean_dice,
    }


# ---------------------------------------------------------------------------
#  Convenience: evaluate from result directory
# ---------------------------------------------------------------------------


def evaluate_result_dir(
    result_dir: str,
    gt_dir: str,
    label_ids: Optional[List[int]] = None,
    label_names: Optional[Dict[int, str]] = None,
    mask_filename: str = "final.nii.gz",
    compute_hd: bool = False,
    save_json: bool = True,
) -> Dict:
    """
    Walk a Search2Adapt result directory, find final pseudo-label masks for
    each case, and compute Dice/ASD against ground-truth labels.

    Expected structure::

        result_dir/
          final_pseudolabels/
            AMOS_CT_0001.nii.gz
            AMOS_CT_0006.nii.gz
            ...

        gt_dir/
          AMOS_CT_0001.nii.gz
          AMOS_CT_0006.nii.gz
          ...

    Args:
        result_dir:     Root of a Search2Adapt run (contains final_pseudolabels/).
        gt_dir:         Directory containing ground-truth NIfTI files.
        label_ids:      Labels to evaluate (default: all non-zero in GT).
        label_names:    label_id → organ_name mapping.
        mask_filename:  Filename inside each case sub-dir (only used when
                        final_pseudolabels/ sub-dir per case; default "final.nii.gz").
        compute_hd:     Whether to compute HD95.
        save_json:      Write gt_metrics.json to result_dir if True.

    Returns:
        {
          "result_dir": str,
          "gt_dir":     str,
          "per_case":   {case_id: <compute_case_metrics() result>, ...},
          "aggregate":  <compute_batch_metrics() result>,
        }
    """
    if label_names is None:
        label_names = AMOS_LABEL_NAMES

    pseudolabel_dir = os.path.join(result_dir, "final_pseudolabels")
    if not os.path.isdir(pseudolabel_dir):
        logger.warning(
            f"final_pseudolabels/ not found under {result_dir}. "
            "Run the pipeline first or call save_final_pseudolabels()."
        )
        return {"error": "final_pseudolabels/ not found", "result_dir": result_dir}

    # Discover prediction files
    pred_files = sorted(
        f
        for f in os.listdir(pseudolabel_dir)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    )
    if not pred_files:
        logger.warning(f"No .nii.gz files found in {pseudolabel_dir}")
        return {"error": "no predictions found", "result_dir": result_dir}

    per_case_results: Dict = {}
    all_case_metrics: List[Dict] = []

    for fname in pred_files:
        # Derive case_id: strip suffix
        case_id = fname.replace(".nii.gz", "").replace(".nii", "")
        pred_path = os.path.join(pseudolabel_dir, fname)

        # Try exact match in gt_dir
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            # Try without channel suffix (_0000)
            alt = case_id.replace("_0000", "") + ".nii.gz"
            gt_path = os.path.join(gt_dir, alt)
        if not os.path.exists(gt_path):
            logger.warning(f"GT not found for {case_id}: {gt_path} — skipping")
            continue

        try:
            cm = compute_case_metrics(
                pred_path=pred_path,
                gt_path=gt_path,
                label_ids=label_ids,
                label_names=label_names,
                compute_hd=compute_hd,
            )
            per_case_results[case_id] = cm
            all_case_metrics.append(cm)
            logger.info(
                f"  {case_id}: mean_dice={cm['mean_dice']:.4f}  "
                f"mean_asd={cm['mean_asd']:.2f}mm"
            )
        except Exception as e:
            logger.error(f"  {case_id}: metrics failed — {e}")
            per_case_results[case_id] = {"error": str(e)}

    aggregate = compute_batch_metrics(all_case_metrics)

    output = {
        "result_dir": result_dir,
        "gt_dir": gt_dir,
        "per_case": per_case_results,
        "aggregate": aggregate,
    }

    if save_json:
        out_path = os.path.join(result_dir, "gt_metrics.json")
        with open(out_path, "w") as f:
            # NaN values cannot be serialised as JSON; replace with None
            json.dump(_nan_safe(output), f, indent=2)
        logger.info(f"GT metrics saved to {out_path}")

    return output


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------


def _nan_safe(obj):
    """Recursively replace float('nan') / float('inf') with None for JSON."""
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_safe(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


def print_metrics_table(metrics_output: Dict) -> None:
    """Pretty-print a metrics dict from evaluate_result_dir() or compute_case_metrics()."""
    per_case = metrics_output.get("per_case", {})
    if not per_case:
        # Single-case result
        _print_single_case(metrics_output)
        return

    agg = metrics_output.get("aggregate", {})
    print("\n" + "=" * 80)
    print(f"GT Metrics Summary  —  {len(per_case)} cases")
    print("=" * 80)

    # Per-organ table
    organ_dice = agg.get("per_organ_mean_dice", {})
    organ_asd = agg.get("per_organ_mean_asd", {})
    all_organs = sorted(organ_dice.keys())

    if all_organs:
        print(f"\n{'Organ':<28} {'Mean Dice':>10} {'Mean ASD (mm)':>14}")
        print("-" * 56)
        for organ in all_organs:
            d = organ_dice.get(organ, float("nan"))
            a = organ_asd.get(organ, float("nan"))
            d_str = f"{d:.4f}" if not np.isnan(d) else "  N/A"
            a_str = f"{a:.2f}" if not np.isnan(a) else "  N/A"
            print(f"  {organ:<26} {d_str:>10} {a_str:>14}")
        print("-" * 56)
        d_overall = agg.get("mean_dice", float("nan"))
        a_overall = agg.get("mean_asd", float("nan"))
        d_str = f"{d_overall:.4f}" if not np.isnan(d_overall) else "  N/A"
        a_str = f"{a_overall:.2f}" if not np.isnan(a_overall) else "  N/A"
        print(f"  {'MEAN':<26} {d_str:>10} {a_str:>14}")

    print("=" * 80 + "\n")


def _print_single_case(cm: Dict) -> None:
    case_id = cm.get("case_id", "?")
    print(f"\n{'=' * 60}")
    print(
        f"Case: {case_id}  —  mean_dice={cm.get('mean_dice', float('nan')):.4f}  "
        f"mean_asd={cm.get('mean_asd', float('nan')):.2f}mm"
    )
    print(f"{'Label':>3} {'Organ':<28} {'GT vox':>8} {'Dice':>8} {'ASD (mm)':>10}")
    print("-" * 62)
    for organ, info in cm.get("per_organ", {}).items():
        lid = info.get("label_id", "?")
        gt_vox = info.get("gt_voxels", 0)
        dice = info.get("dice", float("nan"))
        asd = info.get("asd", float("nan"))
        d_str = f"{dice:.4f}" if not np.isnan(dice) else "  N/A"
        a_str = f"{asd:.2f}" if not np.isnan(asd) else "  N/A"
        print(f"{lid:>3} {organ:<28} {gt_vox:>8} {d_str:>8} {a_str:>10}")
    print(f"{'=' * 60}\n")

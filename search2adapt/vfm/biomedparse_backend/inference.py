"""
BiomedParse v2 — multi-GPU 3D inference.

Vendored from the BiomedParse v2 release (https://github.com/microsoft/BiomedParse,
Apache-2.0) and adapted for use as a Search2Adapt subprocess backend.

Differences from the upstream ``inference.py``:

* the hard-coded dataset / checkpoint paths are replaced by module-level
  configuration that the standalone runner overrides;
* ``utils`` / ``npz_utils`` are imported from this package instead of the
  upstream checkout;
* the hydra configuration directory is resolved from ``BIOMEDPARSE_ROOT``
  instead of the current working directory.
"""

import gc
import json
import os
import time
import traceback
from collections import defaultdict
from multiprocessing import Process, Queue

import hydra
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

from .npz_utils import load_npz
from .utils import process_input, process_output, slice_nms

# ==================== configuration ====================
# GPU configuration
CUDA_DEVICES = ["0"]  # GPU device ids to use
NUM_WORKERS = len(CUDA_DEVICES)  # one worker per GPU

# Data paths
INPUT_DIR = ""
OUTPUT_DIR = ""
CHECKPOINT_PATH = ""

# Save options
SAVE_NII = True  # also write the predictions as NIfTI
SAVE_PROB = False  # also write the probability maps
NII_OUTPUT_DIR = None  # None -> "<OUTPUT_DIR>/nii_predictions"
PROB_OUTPUT_DIR = None  # None -> "<OUTPUT_DIR>/probability_maps"

# Inference options
SLICE_BATCH_SIZE = 8
INTERPOLATE_SIZE = 512
NMS_IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5

# Root of the BiomedParse checkout (model code + hydra configs)
BIOMEDPARSE_ROOT = os.environ.get("BIOMEDPARSE_ROOT", "")
# =======================================================


def _build_prompt_text(text_prompts, filename):
    """Validate label-aligned prompts and build the ``[SEP]``-joined text.

    A mismatch here is a prompt-injection error, not a model error: silently
    continuing would assign masks to the wrong class IDs.
    """
    if not isinstance(text_prompts, dict):
        raise ValueError(f"{filename}: text_prompts must be a mapping")

    prompt_map = {}
    for key, value in text_prompts.items():
        if str(key).strip().lower() == "instance_label":
            continue
        try:
            label_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{filename}: prompt key is not a label ID: {key!r}"
            ) from exc
        if label_id in prompt_map:
            raise ValueError(f"{filename}: duplicate prompt label ID {label_id}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{filename}: empty prompt for label {label_id}")
        if "[SEP]" in value.upper():
            raise ValueError(
                f"{filename}: prompt for label {label_id} contains a [SEP] separator"
            )
        prompt_map[label_id] = value.strip()

    if not prompt_map:
        raise ValueError(f"{filename}: no non-background prompts found")

    ids = sorted(prompt_map)
    text = "[SEP]".join(prompt_map[label_id] for label_id in ids)
    if text.count("[SEP]") + 1 != len(ids):
        raise ValueError(
            f"{filename}: prompt count does not match the number of labels"
        )
    return ids, text


def load_case(file_path):
    data = load_npz(file_path)
    image = data["imgs"]
    text_prompts = data["text_prompts"]
    gt = data.get("gts")
    return image, text_prompts, gt


def merge_multiclass_masks(masks, ids):
    bg_mask = 0.5 * torch.ones_like(masks[0:1])
    keep_masks = torch.cat([bg_mask, masks], dim=0)
    class_mask = keep_masks.argmax(dim=0)

    id_map = {j + 1: int(ids[j]) for j in range(len(ids)) if j + 1 != int(ids[j])}
    if len(id_map) > 0:
        orig_mask = class_mask.clone()
        for j in id_map:
            class_mask[orig_mask == j] = id_map[j]

    return class_mask


def postprocess(
    model_outputs, object_existence, threshold=None, do_nms=True
):
    if threshold is None:
        threshold = SCORE_THRESHOLD
    if do_nms and model_outputs.shape[0] > 1:
        return slice_nms(
            model_outputs.sigmoid(),
            object_existence.sigmoid(),
            iou_threshold=NMS_IOU_THRESHOLD,
            score_threshold=threshold,
        )
    mask = (model_outputs.sigmoid()) * (
        object_existence.sigmoid() > threshold
    ).int().unsqueeze(-1).unsqueeze(-1)
    return mask


def compute_dice_coefficient(mask_gt, mask_pred):
    """Dice coefficient between two binary masks."""
    if torch.is_tensor(mask_gt):
        mask_gt = mask_gt.cpu().numpy()
    if torch.is_tensor(mask_pred):
        mask_pred = mask_pred.cpu().numpy()

    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)

    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


def compute_dice_per_class(gt_mask, pred_mask, class_ids):
    """Dice coefficient for every requested class."""
    dice_scores = {}

    for class_id in class_ids:
        gt_class_mask = gt_mask == class_id
        pred_class_mask = pred_mask == class_id
        dice_scores[class_id] = compute_dice_coefficient(gt_class_mask, pred_class_mask)

    return dice_scores


def save_as_nii(mask_array, output_path, affine=None):
    """Save a label array as NIfTI."""
    if torch.is_tensor(mask_array):
        mask_array = mask_array.cpu().numpy()

    if affine is None:
        affine = np.eye(4)

    mask_array = mask_array.astype(np.int16)
    nii_img = nib.Nifti1Image(mask_array, affine)
    nib.save(nii_img, output_path)


def save_probability_maps(prob_maps, output_path, class_ids):
    """Save the per-class probability maps as a compressed NPZ."""
    if torch.is_tensor(prob_maps):
        prob_maps = prob_maps.cpu().numpy()

    prob_maps = prob_maps.astype(np.float16)

    np.savez_compressed(
        output_path, probability_maps=prob_maps, class_ids=np.array(class_ids)
    )


def _hydra_config_dir():
    """Absolute path of the hydra config directory holding ``biomedparse_3D``."""
    root = BIOMEDPARSE_ROOT or os.environ.get("BIOMEDPARSE_ROOT", "")
    if root:
        return os.path.abspath(os.path.join(root, "configs", "model"))
    return os.path.abspath(os.path.join(".", "configs", "model"))


def worker_process(
    gpu_id,
    file_queue,
    result_queue,
    progress_queue,
    input_dir,
    output_dir,
    checkpoint_path,
    nii_dir,
    prob_dir,
    settings=None,
):
    """Worker process: runs the model on one GPU.

    ``settings`` carries the inference options chosen on the command line.
    They cannot travel as module globals: this module is re-imported from
    scratch in every spawned worker, so anything ``main()`` assigned to a
    global would silently fall back to the module defaults (which is how a
    ``--interpolate_size 256`` request used to run at 512, and how the hydra
    config directory ended up resolved against the caller's cwd).
    """
    global SLICE_BATCH_SIZE, INTERPOLATE_SIZE
    global NMS_IOU_THRESHOLD, SCORE_THRESHOLD
    global SAVE_NII, SAVE_PROB, BIOMEDPARSE_ROOT

    if settings:
        SLICE_BATCH_SIZE = settings["slice_batch_size"]
        INTERPOLATE_SIZE = settings["interpolate_size"]
        NMS_IOU_THRESHOLD = settings["nms_threshold"]
        SCORE_THRESHOLD = settings["score_threshold"]
        SAVE_NII = settings["save_nii"]
        SAVE_PROB = settings["save_prob"]
        BIOMEDPARSE_ROOT = settings["biomedparse_root"]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES is remapped to :0

    try:
        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(
            config_dir=_hydra_config_dir(), job_name=f"prediction_gpu_{gpu_id}"
        )
        cfg = hydra.compose(config_name="biomedparse_3D")
        model = hydra.utils.instantiate(cfg, _convert_="object")
        model.load_pretrained(checkpoint_path)

        model.to(device)
        model.eval()

        while True:
            try:
                file = file_queue.get(timeout=5)
                if file is None:
                    break

                file_path = os.path.join(input_dir, file)

                result = process_single_file(
                    file_path,
                    file,
                    output_dir,
                    nii_dir,
                    prob_dir,
                    model,
                    device,
                    gpu_id,
                )

                if result:
                    result_queue.put(result)

                progress_queue.put(
                    {"gpu_id": gpu_id, "file": file, "status": "completed"}
                )

                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                progress_queue.put(
                    {"gpu_id": gpu_id, "file": file, "status": "error", "error": str(e)}
                )
                continue

    except Exception as e:
        progress_queue.put(
            {"gpu_id": gpu_id, "status": "worker_error", "error": str(e)}
        )


def process_single_file(
    file_path,
    filename,
    output_dir,
    nii_output_dir,
    prob_output_dir,
    model,
    device,
    gpu_id,
):
    """Run the model on a single NPZ case."""
    try:
        npz_data = load_npz(file_path)
        imgs = npz_data["imgs"]
        text_prompts = npz_data["text_prompts"]

        has_gt = "gts" in npz_data
        gt_masks = npz_data.get("gts")

        ids, text = _build_prompt_text(text_prompts, filename)

        imgs, pad_width, padded_size, valid_axis = process_input(
            imgs, INTERPOLATE_SIZE
        )
        imgs = imgs.to(device).int()

        input_tensor = {
            "image": imgs.unsqueeze(0),
            "text": [text],
        }

        with torch.no_grad():
            output = model(input_tensor, mode="eval", slice_batch_size=SLICE_BATCH_SIZE)

        mask_logits = output["predictions"]["pred_gmasks"]
        object_existence = output["predictions"]["object_existence"]

        if mask_logits.shape[0] != len(ids):
            raise ValueError(
                f"{filename}: model returned {mask_logits.shape[0]} masks "
                f"for {len(ids)} prompts"
            )
        if object_existence.shape[0] != len(ids):
            raise ValueError(
                f"{filename}: model returned {object_existence.shape[0]} existence "
                f"scores for {len(ids)} prompts"
            )

        mask_logits_resized = F.interpolate(
            mask_logits,
            size=(INTERPOLATE_SIZE, INTERPOLATE_SIZE),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        if SAVE_PROB and prob_output_dir:
            prob_maps = (
                mask_logits_resized.sigmoid()
                * object_existence.sigmoid().unsqueeze(-1).unsqueeze(-1)
            )

            prob_maps_list = []
            for i in range(prob_maps.shape[0]):
                prob_maps_list.append(
                    process_output(prob_maps[i], pad_width, padded_size, valid_axis)
                )

            if torch.is_tensor(prob_maps_list[0]):
                prob_maps_original = torch.stack(prob_maps_list, dim=0)
            else:
                prob_maps_original = np.stack(prob_maps_list, axis=0)

            base_filename = os.path.splitext(filename)[0]
            prob_save_path = os.path.join(prob_output_dir, f"{base_filename}_prob.npz")

            try:
                save_probability_maps(prob_maps_original, prob_save_path, ids)
            except Exception:
                pass  # silently ignore probability-map failures

        mask_preds = postprocess(mask_logits_resized, object_existence)
        mask_preds = merge_multiclass_masks(mask_preds, ids)
        mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis)

        save_path = os.path.join(output_dir, filename)
        np.savez_compressed(save_path, segs=mask_preds)

        if SAVE_NII and nii_output_dir:
            base_filename = os.path.splitext(filename)[0]
            nii_save_path = os.path.join(nii_output_dir, f"{base_filename}_pred.nii.gz")

            try:
                save_as_nii(mask_preds, nii_save_path)
            except Exception:
                pass  # silently ignore NIfTI write failures

        if has_gt and gt_masks is not None:
            if torch.is_tensor(mask_preds):
                mask_preds_np = mask_preds.cpu().numpy()
            else:
                mask_preds_np = mask_preds

            dice_scores = compute_dice_per_class(gt_masks, mask_preds_np, ids)
            valid_dice_scores = [
                score for score in dice_scores.values() if not np.isnan(score)
            ]
            mean_dice = np.mean(valid_dice_scores) if valid_dice_scores else np.NaN

            result = {
                "filename": filename,
                "mean_dice": mean_dice,
            }

            for class_id in ids:
                result[f"dice_class_{class_id}"] = dice_scores.get(class_id, np.NaN)

            return result

        return None

    except Exception as e:
        tqdm.write(
            f"[ERROR] process_single_file({filename}): {e}\n{traceback.format_exc()}"
        )
        return None


def main():
    """Run multi-GPU inference over every NPZ file in ``INPUT_DIR``."""
    tqdm.write("=" * 60)
    tqdm.write("Starting Multi-GPU Inference")
    tqdm.write("=" * 60)
    tqdm.write(f"GPU Devices: {CUDA_DEVICES}")
    tqdm.write(f"Number of Workers: {NUM_WORKERS}")
    tqdm.write(f"Input Directory: {INPUT_DIR}")
    tqdm.write(f"Output Directory: {OUTPUT_DIR}")
    tqdm.write(f"Checkpoint: {CHECKPOINT_PATH}")
    tqdm.write(f"SAVE_NII: {SAVE_NII}")
    tqdm.write(f"SAVE_PROB: {SAVE_PROB}")
    tqdm.write("-" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nii_dir = None
    if SAVE_NII:
        if NII_OUTPUT_DIR is None:
            nii_dir = os.path.join(OUTPUT_DIR, "nii_predictions")
        else:
            nii_dir = NII_OUTPUT_DIR
        os.makedirs(nii_dir, exist_ok=True)
        tqdm.write(f"NII Output Directory: {nii_dir}")

    prob_dir = None
    if SAVE_PROB:
        if PROB_OUTPUT_DIR is None:
            prob_dir = os.path.join(OUTPUT_DIR, "probability_maps")
        else:
            prob_dir = PROB_OUTPUT_DIR
        os.makedirs(prob_dir, exist_ok=True)
        tqdm.write(f"Probability Maps Output Directory: {prob_dir}")

    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")]
    tqdm.write(f"\nFound {len(all_files)} files to process")

    if not all_files:
        tqdm.write("No .npz files found in input directory")
        return

    file_queue = Queue()
    result_queue = Queue()
    progress_queue = Queue()

    for file in all_files:
        file_queue.put(file)

    for _ in range(NUM_WORKERS):
        file_queue.put(None)

    processes = []
    # Spawned workers re-import this module, so the options must travel in the
    # Process arguments rather than as module globals.
    worker_settings = {
        "slice_batch_size": SLICE_BATCH_SIZE,
        "interpolate_size": INTERPOLATE_SIZE,
        "nms_threshold": NMS_IOU_THRESHOLD,
        "score_threshold": SCORE_THRESHOLD,
        "save_nii": SAVE_NII,
        "save_prob": SAVE_PROB,
        "biomedparse_root": BIOMEDPARSE_ROOT,
    }

    for i, gpu_id in enumerate(CUDA_DEVICES):
        p = Process(
            target=worker_process,
            args=(
                gpu_id,
                file_queue,
                result_queue,
                progress_queue,
                INPUT_DIR,
                OUTPUT_DIR,
                CHECKPOINT_PATH,
                nii_dir,
                prob_dir,
                worker_settings,
            ),
            name=f"GPU-{gpu_id}-Worker",
        )
        p.start()
        processes.append(p)

    tqdm.write("\nStarting inference...\n")

    all_results = []
    class_dice_accumulator = defaultdict(list)

    with tqdm(total=len(all_files), desc="Processing files", unit="file") as pbar:
        completed = 0
        worker_errors = []
        while completed < len(all_files):
            try:
                if not progress_queue.empty():
                    progress_info = progress_queue.get(timeout=1)
                    status = progress_info.get("status")
                    if status == "completed":
                        pbar.update(1)
                        completed += 1
                        file = progress_info.get("file", "Unknown")
                        gpu = progress_info.get("gpu_id", "Unknown")
                        tqdm.write(f"[GPU {gpu}] Completed: {file}")
                    elif status == "error":
                        pbar.update(1)
                        completed += 1
                        file = progress_info.get("file", "Unknown")
                        gpu = progress_info.get("gpu_id", "Unknown")
                        error = progress_info.get("error", "Unknown error")
                        tqdm.write(f"[GPU {gpu}] Error processing {file}: {error}")
                    elif status == "worker_error":
                        # e.g. the checkpoint failed to load: no file will ever
                        # be reported, so waiting for `completed` would hang.
                        msg = progress_info.get("error", "unknown error")
                        gpu = progress_info.get("gpu_id", "?")
                        worker_errors.append(f"GPU {gpu}: {msg}")
                        tqdm.write(f"[GPU {gpu}] Worker failed: {msg}")
                        break
                elif not any(p.is_alive() for p in processes):
                    # Workers are gone (e.g. killed by the OOM reaper) while
                    # files remain: fail instead of spinning forever.
                    worker_errors.append("all workers exited before finishing")
                    break

                if not result_queue.empty():
                    result = result_queue.get_nowait()
                    if result:
                        all_results.append(result)

                        for key, value in result.items():
                            if key.startswith("dice_class_"):
                                class_id = int(key.split("_")[-1])
                                class_dice_accumulator[class_id].append(value)

                time.sleep(0.1)

            except Exception:
                continue

    for p in processes:
        p.join()

    if completed < len(all_files) and worker_errors:
        raise RuntimeError(
            f"BiomedParse stopped after {completed}/{len(all_files)} files: "
            + "; ".join(worker_errors)
        )

    while not result_queue.empty():
        result = result_queue.get_nowait()
        if result and result not in all_results:
            all_results.append(result)
            for key, value in result.items():
                if key.startswith("dice_class_"):
                    class_id = int(key.split("_")[-1])
                    class_dice_accumulator[class_id].append(value)

    tqdm.write("\n" + "=" * 60)
    tqdm.write("Inference Complete - Computing Statistics")
    tqdm.write("=" * 60)

    if all_results:
        statistics = {
            "overall": {},
            "per_class": {},
            "per_sample": all_results,
            "config": {
                "num_samples": len(all_results),
                "cuda_devices": CUDA_DEVICES,
                "checkpoint": CHECKPOINT_PATH,
                "save_nii": SAVE_NII,
                "save_prob": SAVE_PROB,
            },
        }

        all_mean_scores = [
            score
            for score in [r["mean_dice"] for r in all_results]
            if not np.isnan(score)
        ]
        if all_mean_scores:
            statistics["overall"] = {
                "mean": float(np.mean(all_mean_scores)),
                "std": float(np.std(all_mean_scores)),
                "min": float(np.min(all_mean_scores)),
                "max": float(np.max(all_mean_scores)),
                "median": float(np.median(all_mean_scores)),
            }
        else:
            statistics["overall"] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
            }

        for class_id in sorted(class_dice_accumulator.keys()):
            class_scores = [
                score
                for score in class_dice_accumulator[class_id]
                if not np.isnan(score)
            ]
            if class_scores:
                statistics["per_class"][f"class_{class_id}"] = {
                    "mean": float(np.mean(class_scores)),
                    "std": float(np.std(class_scores)),
                    "min": float(np.min(class_scores)),
                    "max": float(np.max(class_scores)),
                    "median": float(np.median(class_scores)),
                }
            else:
                statistics["per_class"][f"class_{class_id}"] = {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "median": None,
                }

        json_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)

        tqdm.write(f"\nResults saved to: {json_path}")
        tqdm.write(f"Processed {len(all_results)} files with {NUM_WORKERS} GPUs")

        if SAVE_NII:
            tqdm.write(f"NII predictions saved to: {nii_dir}")

        if SAVE_PROB:
            tqdm.write(f"Probability maps saved to: {prob_dir}")

        tqdm.write("\n" + "-" * 60)
        tqdm.write("Overall Statistics:")
        tqdm.write("-" * 60)
        if statistics["overall"]["mean"] is not None:
            tqdm.write(
                f"  Mean Dice:   {statistics['overall']['mean']:.4f} ± {statistics['overall']['std']:.4f}"
            )
            tqdm.write(f"  Median Dice: {statistics['overall']['median']:.4f}")
            tqdm.write(f"  Min Dice:    {statistics['overall']['min']:.4f}")
            tqdm.write(f"  Max Dice:    {statistics['overall']['max']:.4f}")
        else:
            tqdm.write("  No valid Dice scores")

        tqdm.write("\n" + "-" * 60)
        tqdm.write("Per-Class Statistics:")
        tqdm.write("-" * 60)
        for class_name, stats in sorted(statistics["per_class"].items()):
            class_id = class_name.split("_")[1]
            if stats["mean"] is not None:
                tqdm.write(
                    f"  Class {class_id}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                    f"(median: {stats['median']:.4f}, range: [{stats['min']:.4f}, {stats['max']:.4f}])"
                )
            else:
                tqdm.write(f"  Class {class_id}: No valid scores")
    else:
        tqdm.write("\nNo results to save - no files with ground truth were found.")

    tqdm.write("\n" + "=" * 60)
    tqdm.write("All Done!")
    tqdm.write("=" * 60)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()

"""
Search2Adapt Perceptor Agent (Scout)
领域态势感知 — 系统启动时的一次性分析

职责:
  1. 从目标域数据集随机抽取若干 case, 再从每个 case 随机抽取切片
  2. 将切片拼成 N×N 网格图 (N 由 config.perceptor.grid_size 控制)
   3. 对全部 volume 做程序化特征提取 (强度/噪声/前景比等)
      使用多进程并行加速 (multiprocessing.Pool)
  4. 调用 VLM (no-think 模式) 做视觉分析: 模态/部位/伪影/组织特性/纠错
  5. 合并程序化特征 + VLM 分析 → 通过共享 StrategyMemory 写入统一记忆文件
     (memory_<domain>.json); 如无共享 memory 则 fallback 写独立文件

设计原则:
  - Perceptor 只负责感知, 不负责策略
  - 输出核心字段: text_prompts, organ_label_map, domain_context, image_profile
  - 程序化特征为 MONAI transform 选型服务 (强度窗/归一化/spacing重采样/裁剪等)
  - image_profile 基于全部 case 计算 (不仅仅是 VLM 抽样的 case)
  - 共享 StrategyMemory 由外部 (main.py) 创建并注入, 使 Perceptor/Strategist/MCTS
    共享同一 JSON 文件
"""

import json
import logging
import os
import random
import shutil
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

try:
    from PIL import Image
except ImportError:
    Image = None

# TYPE_CHECKING guard avoids circular import at runtime
from typing import TYPE_CHECKING

from search2adapt.agents.base import BaseAgent, LLMCaller, infer_axial_axis
from search2adapt.config import Search2AdaptConfig
from search2adapt.prompts import PROMPT_REGISTRY
from search2adapt.utils import _find_image_dir

if TYPE_CHECKING:
    from search2adapt.memory import StrategyMemory


# =====================================================================
#  Module-level functions for multiprocessing (must be picklable)
# =====================================================================


def _extract_profile_for_file(fpath: str) -> Optional[Dict]:
    """
    Load a single NIfTI file and extract its image profile.
    Top-level function so it can be used with multiprocessing.Pool.

    Returns profile dict, or None on error.
    """
    try:
        obj = nib.load(fpath)
        data = obj.get_fdata()
        case_file = os.path.basename(fpath)
        return _extract_single_profile(data, case_file)
    except Exception:
        return None


def _extract_single_profile(
    data: np.ndarray,
    case_file: str,
) -> Dict:
    """Extract programmatic image features from a single 3D volume."""
    profile: Dict = {"case_file": case_file}

    # --- Intensity statistics ---
    flat = data.flatten().astype(np.float64)
    profile["intensity_min"] = float(np.min(flat))
    profile["intensity_max"] = float(np.max(flat))
    profile["intensity_mean"] = float(np.mean(flat))
    profile["intensity_std"] = float(np.std(flat))

    pcts = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])
    profile["intensity_p01"] = float(pcts[0])
    profile["intensity_p05"] = float(pcts[1])
    profile["intensity_p25"] = float(pcts[2])
    profile["intensity_p50"] = float(pcts[3])
    profile["intensity_p75"] = float(pcts[4])
    profile["intensity_p95"] = float(pcts[5])
    profile["intensity_p99"] = float(pcts[6])

    # Skewness and kurtosis (on 1-99 percentile clipped data)
    clipped = flat[(flat >= pcts[0]) & (flat <= pcts[6])]
    if len(clipped) > 100:
        profile["intensity_skewness"] = float(scipy_skew(clipped))
        profile["intensity_kurtosis"] = float(scipy_kurtosis(clipped))
    else:
        profile["intensity_skewness"] = 0.0
        profile["intensity_kurtosis"] = 0.0

    # --- Foreground ratio (needed before CT detection) ---
    fg_ratio = _compute_foreground_ratio(data)
    profile["foreground_ratio"] = float(fg_ratio)

    # --- Is data preprocessed? ---
    profile["is_preprocessed"] = bool(
        profile["intensity_min"] >= -1 and profile["intensity_max"] <= 256
    )

    # --- CT detection heuristic ---
    raw_ct = profile["intensity_min"] < -500 and profile["intensity_max"] > 200
    preprocessed_ct = (
        profile["is_preprocessed"]
        and profile["intensity_skewness"] > 0.5
        and fg_ratio < 0.6
    )
    profile["likely_ct"] = bool(raw_ct or preprocessed_ct)

    # --- Spatial dimensions ---
    shape = data.shape
    profile["spatial_shape"] = list(shape)
    profile["memory_footprint_mb"] = float(
        np.prod(shape) * data.dtype.itemsize / (1024 * 1024)
    )

    # --- Noise estimation ---
    noise_std, snr = _estimate_noise(data)
    profile["noise_std"] = float(noise_std)
    profile["snr_estimate"] = float(snr)

    # --- Histogram analysis (foreground-only) ---
    num_peaks, dynamic_range_util = _analyze_histogram(data)
    profile["histogram_num_peaks"] = int(num_peaks)
    profile["dynamic_range_utilization"] = float(dynamic_range_util)

    return profile


def _estimate_noise(data: np.ndarray) -> Tuple[float, float]:
    """
    Estimate noise level and SNR.

    Primary: background region std.
    Fallback (preprocessed data with constant background): local foreground
    patch variance. SNR capped at 1000.
    """
    flat = data.flatten().astype(np.float64)
    p05 = np.percentile(flat, 5)
    p95 = np.percentile(flat, 95)

    bg_mask = flat <= p05
    bg_vals = flat[bg_mask]
    fg_mask = flat >= p95
    fg_vals = flat[fg_mask]

    if len(bg_vals) < 10:
        return 0.0, 0.0

    noise_std = float(np.std(bg_vals))

    if noise_std < 1e-6:
        noise_std = _estimate_foreground_noise(data)

    fg_mean = float(np.mean(fg_vals)) if len(fg_vals) > 0 else 0.0

    if noise_std < 1e-6:
        snr = 1000.0
    else:
        snr = float(fg_mean / noise_std)

    snr = min(snr, 1000.0)
    return noise_std, snr


def _estimate_foreground_noise(data: np.ndarray) -> float:
    """
    Estimate noise from local patch variance in foreground regions.
    Samples 200 small 3D patches; returns median local std.
    """
    if data.ndim < 3:
        return 0.0

    threshold = np.percentile(data, 10)
    fg_mask = data > threshold

    patch_size = 5
    local_stds: List[float] = []

    rng = np.random.default_rng(42)
    coords = np.argwhere(fg_mask)
    if len(coords) < 100:
        return 0.0

    n_patches = min(200, len(coords))
    indices = rng.choice(len(coords), size=n_patches, replace=False)

    half = patch_size // 2
    for idx in indices:
        z, y, x = coords[idx]
        z0, z1 = max(0, z - half), min(data.shape[0], z + half + 1)
        y0, y1 = max(0, y - half), min(data.shape[1], y + half + 1)
        x0, x1 = max(0, x - half), min(data.shape[2], x + half + 1)
        patch = data[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size >= 8:
            local_stds.append(float(np.std(patch)))

    if not local_stds:
        return 0.0
    return float(np.median(local_stds))


def _compute_foreground_ratio(data: np.ndarray) -> float:
    """Fraction of voxels above the 5th percentile (likely foreground)."""
    flat = data.flatten().astype(np.float64)
    threshold = np.percentile(flat, 5)
    fg_count = int(np.sum(flat > threshold))
    total = len(flat)
    return fg_count / total if total > 0 else 0.0


def _analyze_histogram(data: np.ndarray) -> Tuple[int, float]:
    """
    Analyze intensity histogram (foreground-only) for:
      - Number of peaks (tissue separation indicator)
      - Dynamic range utilization
    """
    flat = data.flatten().astype(np.float64)

    # Exclude background
    p05 = np.percentile(flat, 5)
    data_min = float(np.min(flat))

    if p05 - data_min < 1.0:
        fg_flat = flat[flat > p05]
    else:
        fg_flat = flat

    if len(fg_flat) < 100:
        fg_flat = flat

    fp01 = np.percentile(fg_flat, 1)
    fp99 = np.percentile(fg_flat, 99)
    clipped = fg_flat[(fg_flat >= fp01) & (fg_flat <= fp99)]

    if len(clipped) < 100:
        return 1, 0.0

    hist, bin_edges = np.histogram(clipped, bins=256)
    hist = hist.astype(np.float64)

    from scipy.ndimage import gaussian_filter1d

    smoothed = gaussian_filter1d(hist, sigma=3)

    threshold = 0.05 * smoothed.max()
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if (
            smoothed[i] > smoothed[i - 1]
            and smoothed[i] > smoothed[i + 1]
            and smoothed[i] > threshold
        ):
            peaks.append(i)

    num_peaks = max(1, len(peaks))

    # Dynamic range utilization — full data
    full_p01 = np.percentile(flat, 1)
    full_p99 = np.percentile(flat, 99)
    data_range = float(full_p99 - full_p01)
    if data.dtype in (np.int16, np.int32):
        dtype_range = float(np.iinfo(data.dtype).max - np.iinfo(data.dtype).min)
    elif data.dtype in (np.float32, np.float64):
        dtype_range = float(np.max(flat) - np.min(flat)) + 1e-8
    else:
        dtype_range = 4096.0

    dynamic_range_util = data_range / (dtype_range + 1e-8)
    dynamic_range_util = min(1.0, max(0.0, dynamic_range_util))

    return num_peaks, dynamic_range_util


class Perceptor(BaseAgent):
    """
    Perceptor (Scout) Agent — 领域态势感知

    整个 Search2Adapt pipeline 的第一步:
      1. scan_and_sample(): 随机抽取 case + 随机抽取切片, 拼 N×N 网格 (用于 VLM)
      2. extract_image_profile(): 对全部 NIfTI 文件做多进程程序化特征提取
      3. analyze_domain(): 调用 VLM (no-think) 做视觉分析
      4. build_perceptor_output(): 整合程序化特征 + VLM 分析
      5. 通过共享 StrategyMemory 写入统一记忆文件 (memory_<domain>.json)
         如无共享 memory 则 fallback 写入独立的 perceptor_memory.json
    """

    def __init__(
        self,
        config: Search2AdaptConfig,
        memory: "Optional[StrategyMemory]" = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name="Perceptor", config=config, logger=logger)
        self.vlm_caller = LLMCaller(config.vlm, self.logger)
        self.grid_size = config.perceptor.grid_size
        self.num_cases = config.perceptor.num_cases
        self.slices_per_case = self.grid_size * self.grid_size  # fill N×N
        self.cleanup_temp = config.perceptor.cleanup_temp
        self.memory = memory  # shared StrategyMemory (injected from outside)

    # =====================================================================
    #  Main entry point
    # =====================================================================

    def run(self, raw_target_organs: List[str]) -> Dict:
        """
        执行完整的领域分析流程

        Args:
            raw_target_organs: 用户输入的原始 organ 名称列表

        Returns:
            perceptor result dict
        """
        self.logger.info(
            f"[Perceptor] Starting domain analysis, targets={raw_target_organs}, "
            f"grid={self.grid_size}x{self.grid_size}, "
            f"num_cases={self.num_cases}"
        )

        # 1. Scan dataset and sample visual grids (for VLM)
        visual_grids, data_info = self.scan_and_sample()
        image_dir = data_info["image_dir"]

        image_profile = self.extract_image_profile(image_dir)

        # 3. Call VLM for visual domain analysis (no-think mode)
        vlm_analysis = self.analyze_domain(
            raw_target_organs, visual_grids, image_profile
        )

        # 4. Build final perceptor output
        plan = self.build_perceptor_output(
            raw_target_organs, vlm_analysis, image_profile, data_info
        )

        # 5. Persist to memory
        if self.memory is not None:
            # Preferred: write into the shared unified StrategyMemory
            self.memory.save_domain_profile(plan)
            plan["memory_path"] = self.memory.unified_file
        else:
            # Fallback: standalone perceptor_memory.json
            memory_path = self._save_plan(plan)
            plan["memory_path"] = memory_path

        # 6. Cleanup temp if configured
        if self.cleanup_temp:
            self._cleanup_temp()

        n_targets = len(plan["text_prompts"])
        self.logger.info(
            f"[Perceptor] Analysis complete: "
            f"{n_targets} targets, "
            f"modality={plan['domain_context'].get('modality', 'Unknown')}"
        )
        return plan

    # =====================================================================
    #  Step 1: Scan and Sample
    # =====================================================================

    def scan_and_sample(self) -> Tuple[List[Dict], Dict]:
        """
        从 nnUNet 数据集随机抽 case, 每个 case 随机抽切片, 拼 N×N 网格.
        仅用于 VLM 视觉分析; 程序化特征提取独立处理全部 case.

        Returns:
            visual_grids: [{case_file, grid_image_path, num_total_slices, ...}, ...]
            data_info: {total_cases, sampled_cases, image_dir}
        """
        if Image is None:
            raise ImportError("Pillow is required for Perceptor visual sampling")

        data_path = self.config.target_data_path
        image_dir = _find_image_dir(data_path)

        nii_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".nii.gz", ".nii"))]
        )
        if not nii_files:
            raise FileNotFoundError(f"No NIfTI files found in {image_dir}")

        sampled_files = random.sample(nii_files, min(self.num_cases, len(nii_files)))
        temp_dir = os.path.join(self.config.result_dir, "temp", "perceptor_grids")
        os.makedirs(temp_dir, exist_ok=True)

        visual_grids = []

        for idx, fname in enumerate(sampled_files):
            fpath = os.path.join(image_dir, fname)
            try:
                obj = nib.load(fpath)
                data = obj.get_fdata()
                shape = data.shape

                axial_axis = infer_axial_axis(shape)
                total_slices = shape[axial_axis]

                # Randomly pick slice indices
                needed = self.slices_per_case  # grid_size^2
                if total_slices >= needed:
                    indices = sorted(random.sample(range(total_slices), needed))
                else:
                    # Not enough slices — take all, rest will be black
                    indices = list(range(total_slices))

                # Extract 2D slices
                pil_images = []
                for si in indices:
                    slc = [slice(None)] * 3
                    slc[axial_axis] = si
                    slice_2d = data[tuple(slc)]
                    pil_images.append(self._normalize_to_pil(slice_2d))

                # Build N×N grid (pad with black if < needed)
                grid_path = os.path.join(
                    temp_dir, f"grid_{idx:02d}_{fname.replace('.nii.gz', '')}.png"
                )
                self._build_grid(pil_images, grid_path)

                visual_grids.append(
                    {
                        "case_file": fname,
                        "grid_image_path": grid_path,
                        "num_total_slices": total_slices,
                        "sampled_indices": indices,
                        "volume_shape": list(shape),
                    }
                )

                self.logger.info(
                    f"[Perceptor] Case {fname}: shape={shape}, "
                    f"slices={total_slices}, sampled={len(indices)}/{needed}"
                )
            except Exception as e:
                self.logger.warning(f"[Perceptor] Failed to load {fname}: {e}")

        data_info = {
            "total_cases": len(nii_files),
            "sampled_cases": len(visual_grids),
            "image_dir": image_dir,
        }

        self.logger.info(
            f"[Perceptor] Scanned {len(nii_files)} cases, "
            f"sampled {len(visual_grids)} grids ({self.grid_size}x{self.grid_size})"
        )
        return visual_grids, data_info

    # =====================================================================
    #  Image Processing Helpers
    # =====================================================================

    def _normalize_to_pil(self, slice_2d: np.ndarray) -> "Image.Image":
        """Normalize a 2D slice to grayscale PIL Image, upscale if too small."""
        img = slice_2d.astype(np.float32)
        p_low, p_high = np.percentile(img, [1, 99])
        img = np.clip(img, p_low, p_high)
        denom = img.max() - img.min()
        if denom < 1e-6:
            img = np.zeros_like(img, dtype=np.uint8)
        else:
            img = ((img - img.min()) / denom * 255).astype(np.uint8)

        pil_img = Image.fromarray(img, mode="L").convert("RGB")

        # Upscale if too small for visibility
        min_side = 256
        if pil_img.size[0] < min_side or pil_img.size[1] < min_side:
            scale = max(min_side / pil_img.size[0], min_side / pil_img.size[1])
            new_size = (int(pil_img.size[0] * scale), int(pil_img.size[1] * scale))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

        return pil_img

    def _build_grid(self, pil_images: List["Image.Image"], save_path: str):
        """
        Build an N×N grid from pil_images, pad with black if fewer than N*N.
        Save to save_path as PNG.
        """
        n = self.grid_size
        total_cells = n * n

        # Determine cell size: use the first image's size, or default
        if pil_images:
            cell_w, cell_h = pil_images[0].size
        else:
            cell_w, cell_h = 256, 256

        # Resize all images to same cell size
        cells = []
        for img in pil_images:
            cells.append(img.resize((cell_w, cell_h), Image.LANCZOS))

        # Pad with black cells if not enough slices
        while len(cells) < total_cells:
            cells.append(Image.new("RGB", (cell_w, cell_h), (0, 0, 0)))

        # Compose grid
        grid = Image.new("RGB", (cell_w * n, cell_h * n))
        for i, cell in enumerate(cells[:total_cells]):
            row, col = divmod(i, n)
            grid.paste(cell, (col * cell_w, row * cell_h))

        grid.save(save_path, format="PNG")

    # =====================================================================
    #  Step 2: Programmatic Feature Extraction (ALL cases, multiprocessing)
    # =====================================================================

    def extract_image_profile(self, image_dir: str) -> Dict:
        """
        从 image_dir 中全部 NIfTI 文件提取程序化影像特征.
        使用 multiprocessing.Pool 多核加速.

        Args:
            image_dir: directory containing NIfTI files (imagesTr/)

        Returns:
            Aggregated image_profile dict across ALL cases.
        """
        nii_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".nii.gz", ".nii"))]
        )
        if not nii_files:
            self.logger.warning("[Perceptor] No NIfTI files for feature extraction")
            return {}

        fpaths = [os.path.join(image_dir, f) for f in nii_files]
        n_files = len(fpaths)

        # Use up to half of available cores (IO-bound + CPU-bound mix)
        n_workers = min(max(1, cpu_count() // 2), n_files)

        self.logger.info(
            f"[Perceptor] Extracting image profile from {n_files} cases "
            f"using {n_workers} workers..."
        )

        import time as _time

        t0 = _time.time()

        with Pool(processes=n_workers) as pool:
            results = pool.map(_extract_profile_for_file, fpaths)

        # Filter out failed cases
        profiles = [r for r in results if r is not None]
        n_failed = n_files - len(profiles)
        elapsed = _time.time() - t0

        if n_failed > 0:
            self.logger.warning(
                f"[Perceptor] {n_failed}/{n_files} cases failed feature extraction"
            )

        self.logger.info(
            f"[Perceptor] Image profile extracted from {len(profiles)}/{n_files} cases "
            f"in {elapsed:.1f}s ({n_workers} workers)"
        )

        if not profiles:
            return {}

        return self._aggregate_profiles(profiles)

    def _aggregate_profiles(self, profiles: List[Dict]) -> Dict:
        """
        Aggregate per-case profiles into a single image_profile.
        Numeric values: compute mean and range across cases.
        List/categorical values: collect unique values.
        """
        n = len(profiles)
        if n == 0:
            return {}

        # Numeric fields to aggregate with mean
        numeric_fields = [
            "intensity_min",
            "intensity_max",
            "intensity_mean",
            "intensity_std",
            "intensity_p01",
            "intensity_p05",
            "intensity_p25",
            "intensity_p50",
            "intensity_p75",
            "intensity_p95",
            "intensity_p99",
            "intensity_skewness",
            "intensity_kurtosis",
            "noise_std",
            "snr_estimate",
            "foreground_ratio",
            "histogram_num_peaks",
            "dynamic_range_utilization",
            "memory_footprint_mb",
        ]

        agg: Dict = {}

        # Aggregate numeric fields
        for field in numeric_fields:
            vals = [p[field] for p in profiles if field in p]
            if vals:
                agg[field] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                }

        # Likely CT: any case looks like CT?
        ct_flags = [p.get("likely_ct", False) for p in profiles]
        agg["likely_ct"] = any(ct_flags)

        # Noise level classification (based on SNR)
        snr_vals = [p.get("snr_estimate", 0) for p in profiles]
        mean_snr = float(np.mean(snr_vals)) if snr_vals else 0
        if mean_snr > 50:
            agg["noise_level_auto"] = "Low"
        elif mean_snr > 15:
            agg["noise_level_auto"] = "Medium"
        else:
            agg["noise_level_auto"] = "High"

        return agg

    # =====================================================================
    #  Step 3: VLM Domain Analysis (no-think mode)
    # =====================================================================

    def analyze_domain(
        self,
        raw_target_organs: List[str],
        visual_grids: List[Dict],
        image_profile: Dict,
    ) -> Dict:
        """
        Call VLM in no-think mode to analyze domain characteristics.
        Sends grid images + programmatic stats for VLM inspection.
        """
        system_prompt = PROMPT_REGISTRY["PERCEPTOR_ANALYSIS"]

        # Collect grid image paths
        grid_paths = [g["grid_image_path"] for g in visual_grids]

        # Build user prompt with case info + programmatic stats
        case_info_lines = []
        for g in visual_grids:
            case_info_lines.append(
                f"- {g['case_file']}: shape={g['volume_shape']}, "
                f"total_slices={g['num_total_slices']}, "
                f"sampled={len(g['sampled_indices'])}/{self.grid_size * self.grid_size}"
            )

        # Format key programmatic stats for VLM context
        stats_lines = self._format_stats_for_vlm(image_profile)

        user_prompt = (
            f"User Input Targets: {raw_target_organs}\n\n"
            f"Number of sampled cases: {len(visual_grids)}\n"
            f"Grid layout: {self.grid_size}x{self.grid_size} "
            f"(each image shows {self.grid_size * self.grid_size} "
            f"randomly sampled axial slices)\n\n"
            f"Case details:\n" + "\n".join(case_info_lines) + "\n\n"
            f"Programmatic Image Statistics:\n{stats_lines}\n\n"
            "Please analyze the sample images and provide your assessment."
        )

        try:
            # Limit images sent to VLM (context limits)
            max_images = min(len(grid_paths), 5)
            response = self.vlm_caller.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=grid_paths[:max_images],
                temperature=0.1,
                no_think=True,  # Perceptor uses no-think mode
            )

            analysis = self._parse_json_response(response)
            self.logger.info("[Perceptor] VLM analysis successful")

        except Exception as e:
            self.logger.warning(f"[Perceptor] VLM analysis failed: {e}, using fallback")
            analysis = self._fallback_analysis(raw_target_organs)

        return analysis

    def _format_stats_for_vlm(self, profile: Dict) -> str:
        """Format key programmatic stats as readable text for VLM context."""
        lines = []

        if "intensity_min" in profile:
            im = profile["intensity_min"]
            ix = profile["intensity_max"]
            lines.append(
                f"- Intensity range: [{im['mean']:.0f}, {ix['mean']:.0f}] (mean across cases)"
            )

        if "intensity_p50" in profile:
            lines.append(f"- Median intensity: {profile['intensity_p50']['mean']:.0f}")

        if "likely_ct" in profile:
            lines.append(f"- Likely CT (based on HU range): {profile['likely_ct']}")

        if "noise_level_auto" in profile:
            lines.append(f"- Estimated noise level: {profile['noise_level_auto']}")

        if "snr_estimate" in profile:
            lines.append(f"- Estimated SNR: {profile['snr_estimate']['mean']:.1f}")

        if "foreground_ratio" in profile:
            lines.append(
                f"- Foreground ratio: {profile['foreground_ratio']['mean']:.2%}"
            )

        if "histogram_num_peaks" in profile:
            lines.append(
                f"- Histogram peaks: {profile['histogram_num_peaks']['mean']:.0f}"
            )

        return "\n".join(lines) if lines else "(no stats available)"

    def _fallback_analysis(self, raw_target_organs: List[str]) -> Dict:
        """Fallback when VLM is unavailable."""
        return {
            "corrected_targets": raw_target_organs,
            "modality": "Unknown",
            "anatomical_site": "Unknown",
            "contrast_quality": 5,
            "noise_assessment": "Medium",
            "artifacts": ["None"],
            "organ_pairs": [],
            "small_targets": [],
        }

    # =====================================================================
    #  Per-Case Profile Extraction (for Strategist per-case injection)
    # =====================================================================

    def get_case_profile(self, image_path: str) -> Dict:
        """
        Extract programmatic image profile for a SINGLE case.

        This is the per-case counterpart to extract_image_profile() which
        aggregates across all cases. Used by Strategist to calibrate tool
        parameters for each individual case.

        Args:
            image_path: Path to a NIfTI image file.

        Returns:
            Per-case profile dict with the same keys as _extract_single_profile().
            Returns {} on error.
        """
        try:
            result = _extract_profile_for_file(image_path)
            return result or {}
        except Exception as e:
            self.logger.warning(
                f"[Perceptor] get_case_profile failed for {image_path}: {e}"
            )
            return {}

    # =====================================================================
    #  Step 4: Build Perceptor Output
    # =====================================================================

    # dataset.json label spellings are not uniform across prepared datasets
    # (AMOS uses "arota"/"postcava", callers say "aortic"/"vena cava"), so map
    # both sides onto one canonical key instead of silently assigning free IDs.
    _ORGAN_ALIASES = {
        "arota": "aorta",
        "aortic": "aorta",
        "postcava": "inferiorvenacava",
        "venacava": "inferiorvenacava",
        "inferiorvenacava": "inferiorvenacava",
        "rightadrenal": "rightadrenal",
        "rightadrenalgland": "rightadrenal",
        "leftadrenal": "leftadrenal",
        "leftadrenalgland": "leftadrenal",
        "prostateuterus": "prostateuterus",
        "prostateoruterus": "prostateuterus",
    }

    # VFM text-prompt spellings that empirically differ from the short organ
    # name. BiomedParse treats "vena cava" as a near-empty prompt (Dice ~0.00
    # on AMOS-CT) while "inferior vena cava" segments the vessel (Dice ~0.88).
    _VFM_PROMPT_ALIASES = {
        "inferiorvenacava": "inferior vena cava",
        "venacava": "inferior vena cava",
    }

    @classmethod
    def _normalize_organ(cls, name: str) -> str:
        key = "".join(ch for ch in name.lower() if ch.isalnum())
        return cls._ORGAN_ALIASES.get(key, key)

    def _resolve_label_map(self, organs: List[str]) -> Dict[str, int]:
        """Map organ names to the label IDs declared by the dataset.

        Falls back to filling unused IDs for organs the dataset does not name,
        so a typo in the request still yields a usable (if unverified) mapping.
        """
        dataset_labels: Dict[str, int] = {}

        dataset_path = getattr(self.config, "target_data_path", "") or ""
        json_path = os.path.join(dataset_path, "dataset.json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    raw = json.load(f).get("labels", {})
                for name, label_id in raw.items():
                    if str(name).lower() == "background":
                        continue
                    dataset_labels[self._normalize_organ(str(name))] = int(label_id)
            except (OSError, ValueError) as exc:
                self.logger.warning(
                    f"[Perceptor] Could not read label IDs from {json_path}: {exc}; "
                    "falling back to sequential IDs"
                )
        else:
            self.logger.warning(
                f"[Perceptor] No dataset.json at {json_path}; falling back to "
                "sequential label IDs, which may not match labelsTr"
            )

        resolved: Dict[str, int] = {}
        used = set()
        next_free = 1
        for organ in organs:
            key = self._normalize_organ(organ)
            if key in dataset_labels and dataset_labels[key] not in used:
                resolved[organ] = dataset_labels[key]
                used.add(dataset_labels[key])
                continue

            while next_free in used:
                next_free += 1
            resolved[organ] = next_free
            used.add(next_free)
            self.logger.warning(
                f"[Perceptor] '{organ}' is not a label in {json_path}; "
                f"assigned unused ID {next_free}"
            )

        return resolved

    def build_perceptor_output(
        self,
        raw_target_organs: List[str],
        vlm_analysis: Dict,
        image_profile: Dict,
        data_info: Dict,
    ) -> Dict:
        """
        Assemble final perceptor output from VLM analysis + programmatic profile.

        Output structure:
          - text_prompts: {organ_name: VFM text prompt}  ← canonical target list
          - organ_label_map: {organ_name: label_id}
          - domain_context: modality, site, VLM-assessed characteristics
          - image_profile: all programmatic features (for MONAI transform selection)
        """
        corrected = vlm_analysis.get("corrected_targets", raw_target_organs)
        modality = vlm_analysis.get("modality", "Unknown")
        site = vlm_analysis.get("anatomical_site", "Unknown")
        per_organ_site = vlm_analysis.get("per_organ_site", {})

        # Build organ → label_id mapping. The dataset owns these IDs: a
        # sequential renumbering here would disagree with labelsTr, so every
        # downstream mask and Dice score would be attributed to the wrong organ.
        organ_label_map = self._resolve_label_map(corrected)

        # Build text prompts for VFMs (the canonical target dict)
        text_prompts = self._construct_prompts(
            corrected, modality, site, per_organ_site
        )

        # Enrich small_targets with label values
        raw_small = vlm_analysis.get("small_targets", [])
        small_targets = []
        for name in raw_small:
            if name in organ_label_map:
                small_targets.append({"name": name, "label": organ_label_map[name]})
            else:
                # VLM returned a name not in corrected targets — skip
                self.logger.warning(
                    f"[Perceptor] small_target '{name}' not in corrected targets, skipping"
                )

        plan = {
            "timestamp": datetime.now().isoformat(),
            "text_prompts": text_prompts,
            "organ_label_map": organ_label_map,
            "domain_context": {
                "modality": modality,
                "anatomical_site": site,
                "contrast_quality": vlm_analysis.get("contrast_quality", 5),
                "noise_assessment": vlm_analysis.get("noise_assessment", "Medium"),
                "artifacts": vlm_analysis.get("artifacts", ["None"]),
                "organ_pairs": vlm_analysis.get("organ_pairs", []),
                "small_targets": small_targets,
                "per_organ_site": per_organ_site,
            },
            "image_profile": image_profile,
            "data_info": {
                "total_cases": data_info.get("total_cases", 0),
                "sampled_cases": data_info.get("sampled_cases", 0),
            },
        }
        return plan

    def _construct_prompts(
        self,
        corrected_organs: List[str],
        modality: str,
        anatomical_site: str,
        per_organ_site: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Build VFM text prompts: '{organ} in {site} {modality}'.

        Uses per-organ site mapping when available (e.g., for whole-body
        scans where different organs reside in different anatomical regions).
        Falls back to the global anatomical_site otherwise.
        """
        if per_organ_site is None:
            per_organ_site = {}

        prompts = {}
        for organ in corrected_organs:
            # Prefer per-organ site, fall back to global site
            organ_site = per_organ_site.get(organ, anatomical_site)
            term = self._VFM_PROMPT_ALIASES.get(
                self._normalize_organ(organ), organ
            )
            if modality != "Unknown" and organ_site != "Unknown":
                prompts[organ] = f"{term} in {organ_site} {modality}"
            else:
                prompts[organ] = term
        return prompts

    # =====================================================================
    #  Persistence: save plan + memory JSON
    # =====================================================================

    def _save_plan(self, plan: Dict) -> str:
        """Write perceptor results to memory JSON (the only persistent output)."""
        os.makedirs(self.config.result_dir, exist_ok=True)

        memory_path = os.path.join(self.config.result_dir, "perceptor_memory.json")

        # Load existing memory or create new
        memory_data = {"domain_analyses": []}
        if os.path.exists(memory_path):
            try:
                with open(memory_path, "r") as f:
                    memory_data = json.load(f)
            except Exception:
                pass

        # Append this analysis (same structure as the plan itself)
        memory_data["domain_analyses"].append(plan)

        with open(memory_path, "w") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"[Perceptor] Perceptor results saved to: {memory_path}")

        return memory_path

    def _cleanup_temp(self):
        """Remove temp directory if cleanup_temp is True."""
        temp_dir = os.path.join(self.config.result_dir, "temp", "perceptor_grids")
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.info(f"[Perceptor] Cleaned up temp dir: {temp_dir}")

    # =====================================================================
    #  JSON Parsing
    # =====================================================================

    @staticmethod
    def _extract_json_object(text: str) -> str:
        """Extract the outermost JSON object from text using bracket matching."""
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in text: {text[:200]}")

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        raise ValueError(f"Unbalanced braces in JSON: {text[:200]}")

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response (robust to code fences)."""
        clean = response.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:])
        clean = (
            clean.replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )
        clean = self._extract_json_object(clean)
        return json.loads(clean)

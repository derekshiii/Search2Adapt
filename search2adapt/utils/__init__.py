"""
Search2Adapt Utilities
日志配置、数据集扫描、可视化辅助函数

支持:
  - 3D NIfTI数据集扫描
  - 2D PNG/JPG数据集扫描（自动检测）
"""

import logging
import os
from typing import List, Optional, Tuple

# =========================================================================
#  1. 日志配置 (Logging Setup)
# =========================================================================


def setup_logger(
    name: str = "Search2Adapt",
    log_dir: Optional[str] = None,
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """
    Create a configured logger with console + file handlers.

    Args:
        name: logger name
        log_dir: directory for log file (None = no file logging)
        level: logging level (DEBUG, INFO, WARNING, ERROR)
        console: whether to add console handler

    Returns:
        configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-16s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}.log"),
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# =========================================================================
#  2. 数据集扫描 (Dataset Scanning)
# =========================================================================


def scan_nnunet_dataset(
    dataset_path: str,
) -> Tuple[List[str], List[str]]:
    """
    Scan an nnUNet-format dataset directory for image files.

    Looks in: imagesTr/ → imagesTs/ → root directory (fallback)

    Args:
        dataset_path: path to nnUNet dataset root

    Returns:
        (image_paths, case_ids) — sorted lists
    """
    image_dir = _find_image_dir(dataset_path)
    nii_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith((".nii.gz", ".nii"))]
    )

    image_paths = []
    case_ids = []

    for fname in nii_files:
        fpath = os.path.join(image_dir, fname)
        image_paths.append(fpath)

        # Extract case_id: remove _0000 suffix and .nii.gz extension
        case_id = fname
        if case_id.endswith(".nii.gz"):
            case_id = case_id[: -len(".nii.gz")]
        elif case_id.endswith(".nii"):
            case_id = case_id[: -len(".nii")]

        if case_id.endswith("_0000"):
            case_id = case_id[: -len("_0000")]

        case_ids.append(case_id)

    return image_paths, case_ids


def _find_image_dir(data_path: str) -> str:
    """Find image directory in nnUNet dataset (imagesTr > imagesTs > root)"""
    candidates = [
        os.path.join(data_path, "imagesTr"),
        os.path.join(data_path, "imagesTs"),
        data_path,
    ]
    for c in candidates:
        if os.path.isdir(c):
            nii = [f for f in os.listdir(c) if f.endswith((".nii.gz", ".nii"))]
            if nii:
                return c
    return data_path


# =========================================================================
#  3. 配置验证辅助 (Config Helpers)
# =========================================================================


def print_config_summary(config) -> str:
    """Format config as readable summary string"""
    lines = [
        "=" * 60,
        "Search2Adapt Configuration",
        "=" * 60,
        f"Experiment:      {config.experiment_name}",
        f"Result Dir:      {config.result_dir}",
        f"Target Data:     {config.target_data_path}",
        f"Target Organs:   {config.target_organs}",
        f"GPUs:            {config.gpus}",
        "",
        "--- LLM ---",
        f"Model:           {config.llm.model}",
        f"Base URL:        {config.llm.base_url}",
        f"Temperature:     {config.llm.temperature}",
        "",
        "--- MCTS ---",
        f"Max Depth:       {config.mcts.max_depth}",
        f"Max Iterations:  {config.mcts.max_iterations}",
        f"Branching:       {config.mcts.branching_factor}",
        f"UCT Weight:      {config.mcts.exploration_weight}",
        f"Reward Thresh:   {config.mcts.reward_threshold}",
        "",
        "--- Distillation ---",
        f"Planner:         {config.distillation.planner}",
        f"Plans ID:        {config.distillation.plans_identifier}",
        f"Pretrained:      {config.distillation.pretrained_weights}",
        "=" * 60,
    ]
    return "\n".join(lines)

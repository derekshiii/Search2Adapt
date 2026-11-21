"""
Search2Adapt Knowledge Distillation Module
MCTS 搜索最优伪标签 → nnUNet 数据集构建 → nnUNet 训练

两个核心职责:
  1. DatasetBuilder: 将 MCTS 输出的 best pseudo-labels 打包为 nnUNet 格式
  2. DistillationRunner: 调用 nnUNet preprocessing + training

对应论文:
  - Phase 3: 知识蒸馏到轻量学生模型 (nnU-Net ResEncUNet)
  - 输入: {case_id: MaskState} (MCTS 最优路径的 pseudo-labels)
  - 输出: 训练好的 nnUNet 模型
"""

import glob
import json
import logging
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from search2adapt.agents.base import Action, MaskState
from search2adapt.config import Search2AdaptConfig

# =========================================================================
#  1. nnUNet Dataset Builder
# =========================================================================


class DatasetBuilder:
    """
    将 MCTS 搜索得到的最优 pseudo-labels 打包为 nnUNet 格式数据集

    nnUNet 数据集结构:
      Dataset{NNN}_{suffix}/
        imagesTr/       ← 原始图像 ({case}_0000.nii.gz)
        labelsTr/       ← pseudo-labels ({case}.nii.gz)
        dataset.json    ← 元数据 (modality, labels, numTraining, ...)
    """

    def __init__(
        self,
        config: Search2AdaptConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.logger = logger or logging.getLogger("DatasetBuilder")

    def build_dataset(
        self,
        mcts_results: Dict[str, Tuple[MaskState, List[Action], float]],
        adaptation_plan: Dict,
        min_reward: float = 0.0,
        ablation_name: str = "full",
    ) -> str:
        """
        Build nnUNet dataset from MCTS search results.

        Args:
            mcts_results: {case_id: (best_state, best_traj, best_reward)} from MCTS
            adaptation_plan: Perceptor output (text_prompts, modality, etc.)
            min_reward: minimum reward threshold to include a case

        Returns:
            path to the constructed nnUNet dataset directory
        """
        self.logger.info("=" * 60)
        self.logger.info("Building nnUNet dataset from MCTS pseudo-labels")
        self.logger.info("=" * 60)

        original_dataset_path = self.config.target_data_path
        nnunet_raw_root = os.path.dirname(os.path.abspath(original_dataset_path))

        file_ext = ".nii.gz"
        channel_names = self._read_channel_names(original_dataset_path)

        # Extract original suffix
        original_dirname = os.path.basename(os.path.abspath(original_dataset_path))
        match = re.match(r"Dataset(\d+)_", original_dirname)
        original_suffix = original_dirname.split("_", 1)[1] if match else "SFUDA_Data"

        dataset_id = self._find_next_dataset_id(nnunet_raw_root)
        if ablation_name and ablation_name != "full":
            new_dataset_name = (
                f"Dataset{dataset_id:03d}_{original_suffix}_{ablation_name}"
            )
        else:
            new_dataset_name = f"Dataset{dataset_id:03d}_{original_suffix}_Search2Adapt"
        output_dir = os.path.join(nnunet_raw_root, new_dataset_name)

        images_dir = os.path.join(output_dir, "imagesTr")
        labels_dir = os.path.join(output_dir, "labelsTr")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Locate original images
        source_dir = self._find_source_image_dir(original_dataset_path)

        # Copy pseudo-labels and images
        target_organs = list(adaptation_plan.get("text_prompts", {}).keys())
        organ_label_map = adaptation_plan.get("organ_label_map", {})

        success_count = 0
        skipped_count = 0
        reward_filtered = 0

        for case_id, result in mcts_results.items():
            if result is None:
                skipped_count += 1
                self.logger.warning(f"Skipping {case_id}: MCTS search failed")
                continue

            best_state, best_traj, best_reward = result

            # Filter by reward threshold
            if best_reward < min_reward:
                reward_filtered += 1
                self.logger.info(
                    f"Skipping {case_id}: reward {best_reward:.3f} < {min_reward}"
                )
                continue

            # Copy pseudo-label with relabeling to sequential labels
            mask_path = best_state.mask_path
            if not os.path.exists(mask_path):
                self.logger.warning(
                    f"Skipping {case_id}: mask not found at {mask_path}"
                )
                skipped_count += 1
                continue

            label_filename = f"{case_id}{file_ext}"
            dst_label = os.path.join(labels_dir, label_filename)
            self._copy_with_relabel(mask_path, dst_label, organ_label_map)

            img_filename = f"{case_id}_0000{file_ext}"
            src_img = os.path.join(source_dir, img_filename) if source_dir else None
            if src_img and os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(images_dir, img_filename))
                success_count += 1
            else:
                if os.path.exists(best_state.image_path):
                    shutil.copy2(
                        best_state.image_path,
                        os.path.join(images_dir, img_filename),
                    )
                    success_count += 1
                else:
                    self.logger.warning(f"Skipping {case_id}: source image not found")
                    os.remove(dst_label)
                    skipped_count += 1

        labels_json = {"background": "0"}
        for i, organ in enumerate(target_organs, 1):
            labels_json[organ] = str(i)

        dataset_json = {
            "channel_names": channel_names,
            "labels": labels_json,
            "numTraining": success_count,
            "file_ending": file_ext,
            "name": new_dataset_name,
        }

        json_path = os.path.join(output_dir, "dataset.json")
        with open(json_path, "w") as f:
            json.dump(dataset_json, f, indent=4)

        self.logger.info(f"Dataset constructed: {output_dir}")
        self.logger.info(
            f"  Cases: {success_count} included, "
            f"{skipped_count} skipped, {reward_filtered} filtered by reward"
        )
        self.logger.info(f"  Dataset ID: {dataset_id}")

        return output_dir

    def _read_channel_names(self, dataset_path: str) -> Dict:
        """Read channel_names from the source dataset.json, if present."""
        json_path = os.path.join(dataset_path, "dataset.json")
        if not os.path.exists(json_path):
            return {"0": "Unknown"}
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("channel_names", {"0": "Unknown"})
        except (json.JSONDecodeError, IOError):
            return {"0": "Unknown"}

    def _find_next_dataset_id(self, nnunet_raw_root: str) -> int:
        """Find next available nnUNet dataset ID"""
        existing_ids = set()
        if os.path.exists(nnunet_raw_root):
            for d in os.listdir(nnunet_raw_root):
                if d.startswith("Dataset"):
                    try:
                        existing_ids.add(int(d[7:10]))
                    except (ValueError, IndexError):
                        pass

        next_id = 1
        while next_id in existing_ids:
            next_id += 1
        return next_id

    def _find_source_image_dir(self, dataset_path: str) -> Optional[str]:
        """Find the source image directory in original dataset"""
        candidates = [
            os.path.join(dataset_path, "imagesTr"),
            os.path.join(dataset_path, "imagesTs"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return None

    def _copy_with_relabel(
        self,
        src_path: str,
        dst_path: str,
        organ_label_map: Dict[str, int],
    ):
        """
        Copy a NIfTI mask with relabeling to the sequential label scheme.
        """
        nii_obj = nib.load(src_path)
        data = nii_obj.get_fdata()
        unique_labels = np.unique(data)

        expected_labels = set(organ_label_map.values())
        found_labels = set(unique_labels) - {0}

        if found_labels == expected_labels or len(found_labels) == 0:
            shutil.copy2(src_path, dst_path)
            return

        if len(found_labels) == 1 and len(organ_label_map) >= 1:
            old_label = list(found_labels)[0]
            first_organ_label = list(organ_label_map.values())[0]
            if old_label != first_organ_label:
                data[data == old_label] = first_organ_label
            new_nii = nib.Nifti1Image(
                data.astype(np.int16), nii_obj.affine, nii_obj.header
            )
            nib.save(new_nii, dst_path)
            self.logger.debug(
                f"Relabeled {os.path.basename(src_path)}: {old_label} -> {first_organ_label}"
            )
            return

        if found_labels.issubset(expected_labels):
            shutil.copy2(src_path, dst_path)
            return

        self.logger.warning(
            f"Label mismatch in {os.path.basename(src_path)}: "
            f"found {sorted(found_labels)}, expected {sorted(expected_labels)}. "
            f"Copying as-is - this may cause nnUNet preprocessing errors."
        )
        shutil.copy2(src_path, dst_path)

    def build_dataset_from_dir(
        self,
        pseudolabels_dir: str,
        original_images_dir: str,
        organ_names: List[str],
        organ_label_map: Optional[Dict[str, int]] = None,
        modality: str = "CT",
        min_reward: Optional[float] = None,
    ) -> str:
        """
        Build nnUNet dataset directly from saved pseudo-labels directory.

        Args:
            pseudolabels_dir: Path to final_pseudolabels/ directory containing .nii.gz files
            original_images_dir: Path to original images directory (e.g., DatasetXXX/imagesTr)
            organ_names: List of organ names for label mapping
            organ_label_map: Optional dict mapping organ name to label ID (defaults to sequential)
            modality: Modality string (e.g., "CT", "MRI")
            min_reward: Optional reward threshold (read from filename if contains _reward_X.XXX)

        Returns:
            path to the constructed nnUNet dataset directory
        """
        self.logger.info("=" * 60)
        self.logger.info("Building nnUNet dataset from directory")
        self.logger.info(f"  Pseudo-labels: {pseudolabels_dir}")
        self.logger.info(f"  Images: {original_images_dir}")
        self.logger.info("=" * 60)

        # Determine dataset location
        nnunet_raw_root = os.path.dirname(os.path.abspath(original_images_dir))

        # Find next available dataset ID
        dataset_id = self._find_next_dataset_id(nnunet_raw_root)
        new_dataset_name = f"Dataset{dataset_id:03d}_SFUDA_Search2Adapt"
        output_dir = os.path.join(nnunet_raw_root, new_dataset_name)

        images_dir = os.path.join(output_dir, "imagesTr")
        labels_dir = os.path.join(output_dir, "labelsTr")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Get all pseudo-label files
        label_files = sorted(glob.glob(os.path.join(pseudolabels_dir, "*.nii.gz")))

        # Build organ_label_map if not provided
        if organ_label_map is None:
            organ_label_map = {name: i + 1 for i, name in enumerate(organ_names)}

        success_count = 0
        skipped_count = 0
        reward_filtered = 0

        for label_path in label_files:
            # Extract case_id from filename (e.g., "AMOS_CT_0035.nii.gz" or "AMOS_CT_0035_reward_0.835.nii.gz")
            basename = os.path.basename(label_path)
            case_id = basename.replace(".nii.gz", "").replace(".png", "")

            # Check for reward in filename
            reward_match = re.search(r"_reward_([0-9.]+)", case_id)
            if reward_match:
                case_reward = float(reward_match.group(1))
                case_id = re.sub(r"_reward_[0-9.]+", "", case_id)

                if min_reward is not None and case_reward < min_reward:
                    reward_filtered += 1
                    self.logger.info(
                        f"Skipping {case_id}: reward {case_reward:.3f} < {min_reward}"
                    )
                    continue

            # Copy pseudo-label with relabeling
            label_filename = basename
            dst_label = os.path.join(labels_dir, label_filename)
            self._copy_with_relabel(label_path, dst_label, organ_label_map)

            # Find and copy corresponding image
            img_filename = f"{case_id}_0000.nii.gz"
            src_img = os.path.join(original_images_dir, img_filename)

            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(images_dir, img_filename))
                success_count += 1
            else:
                self.logger.warning(f"Source image not found for {case_id}: {src_img}")
                os.remove(dst_label)
                skipped_count += 1

        # Write dataset.json (nnUNet requires string values for labels)
        labels_json = {"background": "0"}
        for i, organ in enumerate(organ_names, 1):
            labels_json[organ] = str(i)

        dataset_json = {
            "channel_names": {"0": modality},
            "labels": labels_json,
            "numTraining": success_count,
            "file_ending": ".png" if label_files and label_files[0].endswith(".png") else ".nii.gz",
            "name": new_dataset_name,
        }

        json_path = os.path.join(output_dir, "dataset.json")
        with open(json_path, "w") as f:
            json.dump(dataset_json, f, indent=4)

        self.logger.info(f"Dataset constructed: {output_dir}")
        self.logger.info(
            f"  Cases: {success_count} included, "
            f"{skipped_count} skipped, {reward_filtered} filtered by reward"
        )
        self.logger.info(f"  Dataset ID: {dataset_id}")

        return output_dir


# =========================================================================
#  2. nnUNet Distillation Runner
# =========================================================================


class DistillationRunner:
    """
    nnUNet 知识蒸馏: preprocessing → training

    封装 nnUNet CLI 命令:
      1. nnUNetv2_plan_and_preprocess -d ID -c CONFIG -pl ResEncUNetPlanner
      2. nnUNetv2_train ID CONFIG all -pretrained_weights CKPT -p nnUNetResEncUNetPlans

    直接复用 PARDA 的 FinetuningModule 逻辑, 适配到 Search2Adapt config 体系
    """

    def __init__(
        self,
        config: Search2AdaptConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.dist_config = config.distillation
        self.logger = logger or logging.getLogger("Distillation")

        self._check_environment()

    def _check_environment(self):
        """Check and set required nnUNet environment variables"""
        required_vars = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]

        # Set environment variables from config.paths
        path_mapping = {
            "nnUNet_raw": self.config.paths.nnunet_raw,
            "nnUNet_preprocessed": self.config.paths.nnunet_preprocessed,
            "nnUNet_results": self.config.paths.nnunet_results,
        }

        for var in required_vars:
            if var not in os.environ or not os.environ[var]:
                if path_mapping[var]:
                    os.environ[var] = path_mapping[var]
                    self.logger.info(f"Set {var}={path_mapping[var]}")
                else:
                    self.logger.warning(f"{var} not set and no path configured")

        # Verify all variables are set
        missing = [v for v in required_vars if v not in os.environ or not os.environ[v]]
        if missing:
            self.logger.warning(
                f"Missing nnUNet environment variables: {missing}. "
                "Ensure these are exported before running distillation."
            )
        else:
            self.logger.info("nnUNet environment variables: OK")

    def run_distillation(
        self,
        dataset_path: str,
        pretrained_weights: Optional[str] = None,
        num_gpus: Optional[int] = None,
    ) -> Dict:
        """
        Run full nnUNet distillation pipeline.

        Args:
            dataset_path: path to the nnUNet dataset directory
            pretrained_weights: optional path to source-domain pretrained weights
            num_gpus: number of GPUs to use (defaults to len(config.gpus))

        Returns:
            dict with dataset_id, configuration, training status
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Knowledge Distillation")
        self.logger.info("=" * 60)

        # Extract dataset ID
        dataset_id = self.extract_dataset_id(dataset_path)
        self.logger.info(f"Dataset ID: {dataset_id}")

        # Determine number of GPUs
        if num_gpus is None:
            num_gpus = len(self.config.gpus) if self.config.gpus else 1
        self.logger.info(f"Using {num_gpus} GPU(s)")

        # Step 1: Preprocessing
        configuration = self.run_preprocessing(dataset_id)
        self.logger.info(f"Auto-detected configuration: {configuration}")

        # Step 2: Training
        weights = pretrained_weights or self.dist_config.pretrained_weights
        self.run_training(dataset_id, configuration, weights, num_gpus)

        return {
            "dataset_id": dataset_id,
            "dataset_path": dataset_path,
            "configuration": configuration,
            "pretrained_weights": weights,
            "num_gpus": num_gpus,
            "planner": self.dist_config.planner,
            "plans_identifier": self.dist_config.plans_identifier,
            "status": "completed",
        }

    # =====================================================================
    #  Preprocessing
    # =====================================================================

    def run_preprocessing(self, dataset_id: int) -> str:
        """
        Step 1: nnUNet plan and preprocess.
        Uses the "3d_fullres" configuration.

        Returns:
            configuration string
        """
        auto_config = self._determine_config_from_dataset(dataset_id)

        self.logger.info("-" * 50)
        self.logger.info(f"Preprocessing: Dataset {dataset_id}")
        self.logger.info(f"Configuration: {auto_config}")
        self.logger.info(f"Planner: {self.dist_config.planner}")
        self.logger.info("-" * 50)

        cmd_executable = self._get_cmd_path("nnUNetv2_plan_and_preprocess")

        cmd = [
            cmd_executable,
            "-d",
            str(dataset_id),
            "-c",
            auto_config,
            "-pl",
            self.dist_config.planner,
            "--verify_dataset_integrity",
        ]

        self._execute_command(cmd, "Preprocessing")
        return auto_config

    # =====================================================================
    #  Training
    # =====================================================================

    def run_training(
        self,
        dataset_id: int,
        configuration: str,
        checkpoint_path: Optional[str] = None,
        num_gpus: int = 1,
    ):
        """
        Step 2: nnUNet training with pretrained weights.

        Args:
            dataset_id: nnUNet dataset ID
            configuration: configuration string (e.g., "3d_fullres")
            checkpoint_path: optional path to pretrained weights
            num_gpus: number of GPUs to use for training
        """
        self.logger.info("-" * 50)
        self.logger.info(f"Training: Dataset {dataset_id}")
        self.logger.info(f"Configuration: {configuration}")
        self.logger.info(f"Fold: {self.dist_config.fold}")
        self.logger.info(f"Plans: {self.dist_config.plans_identifier}")
        self.logger.info(f"GPUs: {num_gpus}")
        if checkpoint_path:
            self.logger.info(f"Pretrained weights: {checkpoint_path}")
        self.logger.info("-" * 50)

        cmd_executable = self._get_cmd_path("nnUNetv2_train")

        cmd = [
            cmd_executable,
            str(dataset_id),
            configuration,
            self.dist_config.fold,
            "-p",
            self.dist_config.plans_identifier,
            "-num_gpus",
            str(num_gpus),
        ]

        if checkpoint_path and os.path.exists(checkpoint_path):
            cmd.extend(["-pretrained_weights", checkpoint_path])

        self._execute_command(cmd, "Training")

    # =====================================================================
    #  Utilities
    # =====================================================================

    def extract_dataset_id(self, dataset_path: str) -> int:
        """Extract dataset ID from directory name (e.g., Dataset003_...)"""
        dirname = os.path.basename(os.path.normpath(dataset_path))
        match = re.match(r"Dataset(\d{3})_", dirname)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Could not extract Dataset ID from path: {dataset_path}")

    def _determine_config_from_dataset(self, dataset_id: int) -> str:
        """
        Determine the nnU-Net configuration for a dataset.

        Only 3D NIfTI datasets are produced by this release, so the
        configuration is always "3d_fullres".
        """
        return "3d_fullres"

    def _get_cmd_path(self, command_name: str) -> str:
        """Get command path, checking nnUNet env if configured"""
        env_path = self.config.paths.nnunet_env_path
        if env_path:
            base = env_path.rstrip(os.path.sep)
            if base.endswith("bin"):
                cmd_path = os.path.join(base, command_name)
            else:
                cmd_path = os.path.join(base, "bin", command_name)

            if not os.path.exists(cmd_path):
                self.logger.warning(
                    f"Executable not found at: {cmd_path}, falling back to system PATH"
                )
                return command_name
            return cmd_path
        return command_name

    def _execute_command(self, cmd: List[str], stage_name: str):
        """Execute a subprocess command with live logging"""
        self.logger.info(f"Executing: {' '.join(cmd)}")
        env = os.environ.copy()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
            )
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.logger.info(f"[{stage_name}] {line}")

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

            self.logger.info(f"{stage_name} completed successfully.")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"{stage_name} failed with return code {e.returncode}")
            raise
        except Exception as e:
            self.logger.error(f"{stage_name} execution error: {e}")
            raise

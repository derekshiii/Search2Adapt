"""
Search2Adapt Executor Agent (Actor)
工具执行编排 — MCTS Phase 2.5 (介于 Expansion 和 Evaluation 之间)

职责:
  1. 接收 Strategist 提出的 Action, 执行状态转移 m_t → m_{t+1}
  2. 普通后处理工具: 直接调用 ToolExecutor
  3. VFM 工具: 生成 SAM3 config / BiomedParse npz, 调用推理, 合并结果

对应论文:
  - Executor (Actor): 执行具体动作, 实现状态转移函数 P(s_t, a_t) = s_{t+1}
  - 对于 VFM 类动作, Executor 自主处理配置生成和数据准备

Supports:
  - 3D NIfTI medical images
"""

import json
import logging
import os
import shutil
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from search2adapt.agents.base import Action, BaseAgent, MaskState, infer_axial_axis
from search2adapt.config import Search2AdaptConfig
from search2adapt.toolbox.tools import TOOL_REGISTRY, ToolExecutor


class Executor(BaseAgent):
    """
    Executor (Actor) Agent — 工具执行与状态转移

    Operates on 3D NIfTI medical images.
    """

    def __init__(
        self,
        config: Search2AdaptConfig,
        adaptation_plan: Optional[Dict] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name="Executor", config=config, logger=logger)
        self.tool_executor = ToolExecutor(config, self.logger)

        self.adaptation_plan = adaptation_plan or {}
        self.target_organs: List[str] = list(
            self.adaptation_plan.get("text_prompts", {}).keys()
        )
        self.organ_label_map: Dict[str, int] = self.adaptation_plan.get(
            "organ_label_map", {}
        )
        self.text_prompts: Dict[str, str] = self.adaptation_plan.get("text_prompts", {})

    def set_adaptation_plan(self, plan: Dict):
        """Update domain context from Perceptor output"""
        self.adaptation_plan = plan
        self.target_organs = list(plan.get("text_prompts", {}).keys())
        self.organ_label_map = plan.get("organ_label_map", {})
        self.text_prompts = plan.get("text_prompts", {})

    # =====================================================================
    #  Main entry: execute action → new state
    # =====================================================================

    def run(self, state: MaskState, action: Action) -> MaskState:
        """
        Execute an action and return the new state.

        Args:
            state: current MaskState (image + mask)
            action: Action to execute (tool_id + params)

        Returns:
            new MaskState with updated mask_path and step
        """
        tool_id = action.tool_id
        self.logger.info(
            f"[Executor] case={state.case_id} step={state.step} "
            f"tool={tool_id} params={action.params}"
        )

        # Output path for new mask
        case_dir = os.path.join(self.config.result_dir, state.case_id)
        os.makedirs(case_dir, exist_ok=True)

        ext = ".nii.gz"

        output_path = os.path.join(
            case_dir,
            f"step_{state.step + 1:02d}_{tool_id}{ext}",
        )

        preprocessed_image_path: Optional[str] = None
        if tool_id in ("sam3", "biomedparse"):
            output_path = self._execute_vfm(state, action, output_path)
        elif tool_id == "dual_expert":
            output_path = self._execute_dual_expert(
                state, action, case_dir, output_path
            )
        elif tool_id in (
            "enhance_image",
            "smooth_image",
            "normalize_intensity",
            "clip_intensity",
        ):
            # Preprocessing modifies the image, not the mask
            preprocessed_image_path = self._execute_preprocess(
                state, action, case_dir, tool_id
            )
        else:
            # Standard post-processing tools (postprocess_segmentation,
            # gaussian_smooth_mask, morphological_closing/opening, symmetry_checker)
            output_path = self.tool_executor.execute(
                tool_id=tool_id,
                image_path=state.image_path,
                mask_path=state.mask_path,
                output_path=output_path,
                params=action.params,
            )

        # Diagnostic tools (e.g. symmetry_checker) are non-depth-consuming:
        # they don't modify the mask or advance the step counter.
        tool_info = TOOL_REGISTRY.get(tool_id, {})
        is_diagnostic = tool_info.get("category") == "diagnostic"

        if is_diagnostic:
            new_state = state.clone(
                new_mask_path=state.mask_path,
                new_step=state.step,
            )
        else:
            # Create new state
            new_state = state.clone(
                new_mask_path=output_path,
                new_step=state.step + 1,
            )

        # If preprocessing was applied, update image_path in new state
        if (
            tool_id
            in (
                "enhance_image",
                "smooth_image",
                "normalize_intensity",
                "clip_intensity",
            )
            and preprocessed_image_path is not None
        ):
            new_state.image_path = preprocessed_image_path
            new_state.mask_path = state.mask_path  # mask unchanged

        if is_diagnostic:
            self.logger.info(
                f"[Executor] Completed diagnostic: {tool_id} "
                f"(step unchanged at {state.step})"
            )
        else:
            self.logger.info(
                f"[Executor] Completed: {tool_id} → {os.path.basename(output_path)}"
            )
        return new_state

    # =====================================================================
    #  Image Preprocessing (operates on image, not mask)
    # =====================================================================

    def _execute_preprocess(
        self, state: MaskState, action: Action, case_dir: str, tool_id: str
    ) -> str:
        """Preprocess image (enhance or smooth); mask stays the same.

        Returns:
            Path to the preprocessed image.
        """
        ext = ".nii.gz"
        preprocessed_path = os.path.join(case_dir, f"preprocessed_{tool_id}{ext}")
        self.tool_executor.execute(
            tool_id=tool_id,
            image_path=state.image_path,
            mask_path=state.mask_path,
            output_path=preprocessed_path,
            params=action.params,
        )
        return preprocessed_path

    # =====================================================================
    #  VFM Dispatch
    # =====================================================================

    def _execute_vfm(self, state: MaskState, action: Action, output_path: str) -> str:
        tool_id = action.tool_id

        if tool_id == "sam3":
            return self._execute_sam3(state, action, output_path)
        elif tool_id == "biomedparse":
            return self._execute_biomedparse(state, action, output_path)
        else:
            raise ValueError(f"Unknown VFM tool: {tool_id}")

    # =====================================================================
    #  SAM3 Execution
    # =====================================================================

    def _execute_sam3(self, state: MaskState, action: Action, output_path: str) -> str:
        """
        Execute SAM3 inference.
        Generates config JSON → subprocess call → merge output.
        """
        case_dir = os.path.dirname(output_path)
        sam3_work_dir = os.path.join(case_dir, "sam3_work")
        os.makedirs(sam3_work_dir, exist_ok=True)

        # Determine prompt mode
        use_visual = action.params.get("use_visual_prompts", False)

        if use_visual and state.mask_path and os.path.exists(state.mask_path):
            config = self._generate_sam3_visual_config(state)
        else:
            config = self._generate_sam3_text_config(state)

        # Write config JSON
        config_path = os.path.join(sam3_work_dir, "sam3_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Run SAM3
        sam3_out_dir = os.path.join(sam3_work_dir, "output")
        self.tool_executor.sam3.run(config_path, sam3_out_dir)

        # Find and copy output
        output_mask = self._find_sam3_output(sam3_out_dir, state.case_id)
        if output_mask:
            shutil.copy2(output_mask, output_path)
        elif state.mask_path and os.path.exists(state.mask_path):
            self.logger.warning("SAM3 produced no output, keeping previous mask")
            shutil.copy2(state.mask_path, output_path)
        else:
            raise RuntimeError(
                f"SAM3 produced no output for case {state.case_id} "
                f"and no previous mask exists to fall back on"
            )

        return output_path

    def _generate_sam3_text_config(self, state: MaskState) -> Dict:
        """Generate SAM3 config with text-only prompts"""
        prompts = []
        for organ, label_id in self.organ_label_map.items():
            prompts.append(
                {
                    "obj_id": label_id,
                    "prompt_type": "text",
                    "text": organ,
                }
            )

        case_name = os.path.splitext(
            os.path.basename(state.image_path).replace(".nii.gz", "").replace(".png", "")
        )[0]

        return {
            case_name: {
                "image": os.path.abspath(state.image_path),
                "prompts": prompts,
            }
        }

    def _generate_sam3_visual_config(self, state: MaskState) -> Dict:
        """
        Generate SAM3 config with text + visual prompts.
        Uses existing mask to extract bounding boxes and key slices.
        """
        mask_obj = nib.load(state.mask_path)
        mask_data = np.round(mask_obj.get_fdata()).astype(np.int16)

        prompts = []
        for organ, label_id in self.organ_label_map.items():
            organ_mask = mask_data == label_id
            bbox_info = self._extract_bbox_and_key_slice(organ_mask, mask_data.shape)
            if bbox_info is not None:
                prompts.append(
                    {
                        "obj_id": label_id,
                        "prompt_type": "text_visual",
                        "text": organ,
                        "box": bbox_info["box"],
                        "key_slice": bbox_info["key_slice"],
                    }
                )
            else:
                # No existing mask for this organ — text only
                prompts.append(
                    {
                        "obj_id": label_id,
                        "prompt_type": "text",
                        "text": organ,
                    }
                )

        case_name = os.path.splitext(
            os.path.basename(state.image_path).replace(".nii.gz", "").replace(".png", "")
        )[0]

        return {
            case_name: {
                "image": os.path.abspath(state.image_path),
                "prompts": prompts,
            }
        }

    def _extract_bbox_and_key_slice(
        self, organ_mask: np.ndarray, volume_shape: Tuple
    ) -> Optional[Dict]:
        """
        Extract 2D bounding box and key slice index from 3D binary mask.

        Returns:
            {"box": [x, y, w, h], "key_slice": int} or None if empty mask
        """
        if organ_mask.sum() == 0:
            return None

        axial_axis = infer_axial_axis(volume_shape)
        num_slices = volume_shape[axial_axis]

        # Find key slice (largest organ area)
        slice_areas = []
        for i in range(num_slices):
            slices = [slice(None)] * 3
            slices[axial_axis] = i
            area = int(organ_mask[tuple(slices)].sum())
            slice_areas.append(area)

        key_slice = int(np.argmax(slice_areas))

        # 2D bbox on key slice
        slices = [slice(None)] * 3
        slices[axial_axis] = key_slice
        key_mask = organ_mask[tuple(slices)]

        coords = np.where(key_mask)
        if len(coords[0]) == 0:
            return None

        # coords[0] and coords[1] correspond to remaining_axes
        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())

        box = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

        return {"box": box, "key_slice": key_slice}

    def _find_sam3_output(self, output_dir: str, case_id: str) -> Optional[str]:
        ext = ".nii.gz"
        """Find the SAM3 output mask file, matching by case_id first."""
        if not os.path.isdir(output_dir):
            return None

        # Try exact case_id-based names first (mirrors _find_bp_output logic)
        for suffix in [f"{case_id}_pred{ext}", f"{case_id}{ext}"]:
            candidate = os.path.join(output_dir, suffix)
            if os.path.exists(candidate):
                return candidate

        # Fallback: return any .nii.gz in the directory
        for f in os.listdir(output_dir):
            if f.endswith(ext):
                return os.path.join(output_dir, f)
        return None

    # =====================================================================
    #  BiomedParse Execution
    # =====================================================================

    def _execute_biomedparse(
        self, state: MaskState, action: Action, output_path: str
    ) -> str:
        """
        Execute BiomedParse inference.
        Prepares NPZ data → multiprocessing call → collect output.
        """
        case_dir = os.path.dirname(output_path)
        bp_work_dir = os.path.join(case_dir, "bp_work")
        os.makedirs(bp_work_dir, exist_ok=True)

        # Prepare NPZ input
        npz_dir = os.path.join(bp_work_dir, "input")
        self._prepare_biomedparse_npz(state, npz_dir)

        # Run BiomedParse. ToolboxConfig carries the run-wide inference
        # settings; parameters written into a plan step take precedence.
        bp_out_dir = os.path.join(bp_work_dir, "output")
        bp_params = {
            "slice_batch_size": self.config.toolbox.bp_slice_batch_size,
            "interpolate_size": self.config.toolbox.bp_interpolate_size,
            "nms_threshold": self.config.toolbox.bp_nms_threshold,
            "score_threshold": self.config.toolbox.bp_score_threshold,
            **action.params,
        }
        self.tool_executor.biomedparse.run(
            data_path=npz_dir,
            output_dir=bp_out_dir,
            params=bp_params,
        )

        # Find and copy output
        output_mask = self._find_bp_output(bp_out_dir, state.case_id)
        if output_mask:
            shutil.copy2(output_mask, output_path)
        elif state.mask_path and os.path.exists(state.mask_path):
            self.logger.warning("BiomedParse produced no output, keeping previous mask")
            shutil.copy2(state.mask_path, output_path)
        else:
            raise RuntimeError(
                f"BiomedParse produced no output for case {state.case_id} "
                f"and no previous mask exists to fall back on"
            )

        return output_path

    def _prepare_biomedparse_npz(self, state: MaskState, output_dir: str):
        """
        Convert NIfTI image to BiomedParse NPZ format.

        NPZ structure:
          imgs: uint8 array
          gts: uint8 array (zeros if no ground truth)
          text_prompts: dict with label_id → text prompt
        """
        os.makedirs(output_dir, exist_ok=True)

        img_obj = nib.load(state.image_path)
        img_data = img_obj.get_fdata()

        # Normalize to uint8
        p_low, p_high = np.percentile(img_data, [1, 99])
        img_data = np.clip(img_data, p_low, p_high)
        denom = img_data.max() - img_data.min()
        if denom > 1e-6:
            img_data = ((img_data - img_data.min()) / denom * 255).astype(np.uint8)
        else:
            img_data = np.zeros_like(img_data, dtype=np.uint8)

        # Build text prompts map using real label_ids from organ_label_map
        text_prompts_map = {}
        for organ, prompt in self.text_prompts.items():
            label_id = self.organ_label_map.get(organ, 0)
            text_prompts_map[str(label_id)] = prompt
        text_prompts_map["instance_label"] = 0

        # Placeholder ground truth (zeros)
        gts = np.zeros_like(img_data, dtype=np.uint8)

        # Save NPZ
        case_name = state.case_id
        npz_path = os.path.join(output_dir, f"{case_name}.npz")
        np.savez_compressed(
            npz_path,
            imgs=img_data,
            gts=gts,
            text_prompts=json.dumps(text_prompts_map),
        )
        self.logger.info(f"Prepared BiomedParse NPZ: {npz_path}")

    def _find_bp_output(self, output_dir: str, case_id: str) -> Optional[str]:
        ext = ".nii.gz"
        """Find BiomedParse output mask file (checks nii_predictions/ subdir first)"""
        if not os.path.isdir(output_dir):
            return None

        # BiomedParse saves NIfTI to nii_predictions/ subdirectory
        nii_dir = os.path.join(output_dir, "nii_predictions")
        search_dirs = [nii_dir, output_dir]

        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            # Try exact match first
            for suffix in [f"{case_id}_pred{ext}", f"{case_id}{ext}"]:
                candidate = os.path.join(search_dir, suffix)
                if os.path.exists(candidate):
                    return candidate
            # Fallback: any .nii.gz file
            for f in os.listdir(search_dir):
                if f.endswith(ext):
                    return os.path.join(search_dir, f)
        return None

    # =====================================================================
    #  Dual Expert: BiomedParse → SAM3 fused mask (as a plan step)
    # =====================================================================

    def _execute_dual_expert(
        self,
        state: MaskState,
        action: Action,
        case_dir: str,
        output_path: str,
    ) -> str:
        self.logger.info("[Executor] dual_expert: running BiomedParse + SAM3 fusion")

        bp_action = Action(
            tool_id="biomedparse",
            params=action.params,
            description="dual_expert sub-step: BiomedParse",
        )
        ext = ".nii.gz"
        bp_out = os.path.join(case_dir, f"dual_expert_bp_{state.step:02d}{ext}")
        bp_out = self._execute_vfm(state, bp_action, bp_out)

        sam3_action = Action(
            tool_id="sam3",
            params={"use_visual_prompts": False, **action.params},
            description="dual_expert sub-step: SAM3",
        )
        sam3_out = os.path.join(case_dir, f"dual_expert_sam3_{state.step:02d}{ext}")
        sam3_out = self._execute_vfm(state, sam3_action, sam3_out)

        self._fuse_dual_masks(bp_out, sam3_out, output_path)
        self.logger.info(
            f"[Executor] dual_expert fusion complete → {os.path.basename(output_path)}"
        )
        return output_path

    # =====================================================================
    #  Initial Pseudo-label Generation (bootstrapping)
    # =====================================================================

    def generate_initial_pseudolabel(
        self,
        image_path: str,
        case_id: str,
        method: str = "biomedparse",
        _is_fallback: bool = False,
    ) -> MaskState:
        """
        Generate initial pseudo-label for a case (before MCTS search).
        This is step 0 — creating the root state.

        If the primary VFM fails, automatically tries the alternative VFM
        as a fallback (unless already in fallback mode).

        Args:
            image_path: path to NIfTI image
            case_id: case identifier
            method: "biomedparse", "sam3", or "dual"
            _is_fallback: internal flag to prevent infinite recursion

        Returns:
            MaskState at step=0 with initial pseudo-label
        """
        case_dir = os.path.join(self.config.result_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)

        ext = ".nii.gz"
        output_path = os.path.join(case_dir, f"step_00_initial{ext}")

        state = MaskState(
            image_path=image_path,
            mask_path="",
            case_id=case_id,
            step=0,
        )

        if method == "biomedparse":
            tool_id = "biomedparse"
            action = Action(
                tool_id=tool_id,
                params={},
                description=f"Initial pseudo-label via {tool_id}",
            )
            try:
                new_state = self.run(state, action)
                if new_state.mask_path and os.path.exists(new_state.mask_path):
                    shutil.move(new_state.mask_path, output_path)
                    new_state.mask_path = output_path
                new_state.step = 0
                return new_state
            except Exception as e:
                if _is_fallback:
                    raise RuntimeError(
                        f"Both VFMs failed for initial pseudo-label of {case_id}"
                    ) from e
                self.logger.warning(
                    f"[Executor] BiomedParse initial failed for {case_id}: {e}, "
                    f"falling back to SAM3"
                )
                return self.generate_initial_pseudolabel(
                    image_path, case_id, method="sam3", _is_fallback=True
                )

        elif method == "sam3":
            tool_id = "sam3"
            action = Action(
                tool_id=tool_id,
                params={"use_visual_prompts": False},
                description=f"Initial pseudo-label via {tool_id}",
            )
            try:
                new_state = self.run(state, action)
                if new_state.mask_path and os.path.exists(new_state.mask_path):
                    shutil.move(new_state.mask_path, output_path)
                    new_state.mask_path = output_path
                new_state.step = 0
                return new_state
            except Exception as e:
                if _is_fallback:
                    raise RuntimeError(
                        f"Both VFMs failed for initial pseudo-label of {case_id}"
                    ) from e
                self.logger.warning(
                    f"[Executor] SAM3 initial failed for {case_id}: {e}, "
                    f"falling back to BiomedParse"
                )
                return self.generate_initial_pseudolabel(
                    image_path, case_id, method="biomedparse", _is_fallback=True
                )

        elif method == "dual":
            bp_state = self.generate_initial_pseudolabel(
                image_path, case_id, method="biomedparse"
            )
            sam3_state = self.generate_initial_pseudolabel(
                image_path, case_id, method="sam3"
            )
            fused_path = os.path.join(case_dir, f"step_00_fused{ext}")
            self._fuse_dual_masks(
                bp_state.mask_path, sam3_state.mask_path, fused_path
            )
            return MaskState(
                image_path=image_path,
                mask_path=fused_path,
                case_id=case_id,
                step=0,
            )
        else:
            raise ValueError(f"Unknown initial method: {method}")

    def _fuse_dual_masks(self, bp_path: str, sam3_path: str, output_path: str):
        """Fuse BiomedParse and SAM3 masks (BP priority, fill with SAM3)"""
        bp_obj = nib.load(bp_path)
        sam3_obj = nib.load(sam3_path)
        bp_data = np.round(bp_obj.get_fdata()).astype(np.int16)
        sam3_data = np.round(sam3_obj.get_fdata()).astype(np.int16)

        fused = bp_data.copy()
        num_labels = max(int(bp_data.max()), int(sam3_data.max()))

        for label_id in range(1, num_labels + 1):
            sam3_region = sam3_data == label_id
            bp_occupied = fused > 0
            additional = sam3_region & (~bp_occupied)
            fused[additional] = label_id

        nib.save(nib.Nifti1Image(fused, bp_obj.affine), output_path)
        self.logger.info(f"Fused dual masks → {os.path.basename(output_path)}")

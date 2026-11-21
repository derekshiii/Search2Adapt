"""
Search2Adapt Evaluator Agent (Critic)
LLM 驱动的 Reward 函数 — MCTS Phase 3

职责:
  1. 接收 Executor 产出的中间态 mask, 提取几何特征
  2. (Text Mode) 将特征文本化后交给 LLM 打分
  3. (Visual Mode) 生成 2x2 overlay grid 交给 VLM 打分
  4. (Plan Mode) 纯文本评估计划质量 — CoTS Phase 1 Critic 角色
  5. (Viability Mode) 计划执行中途可行性检查 — CoTS Phase 2
  6. 返回 EvalResult (reward, scores, decision, defects)
  7. 将每次评估结果写入 StrategyMemory (eval_records section)

对应论文:
  - Reward Function R(s_t, a_t): 非模拟, 基于 LLM 解剖学先验
  - CoTS Critic: 评估计划质量 + 提供改进建议
  - 四维评分 (text/visual): 解剖合理性 + 拓扑/边界 + 平滑/形态 + 体积/误检
  - 四维评分 (plan_eval): 缺陷覆盖 + 步骤连贯 + 工具适当 + 预期效果

Score Weight Sets:
  Text mode:   anatomical_plausibility(0.35) + topological_integrity(0.25)
               + boundary_smoothness(0.20) + volumetric_consistency(0.20)
  Visual mode: anatomical_plausibility(0.35) + boundary_precision(0.25)
               + morphological_consistency(0.20) + false_positives(0.20)
  Plan mode:   defect_coverage(0.30) + step_coherence(0.25)
               + tool_appropriateness(0.25) + expected_effectiveness(0.20)
"""

import base64
import json
import logging
import os
import random

import numpy as np

try:
    from json_repair import repair_json as _repair_json

    def _loads_robust(text: str) -> dict:
        """Parse JSON, falling back to json_repair on syntax errors."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = _repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
            return json.loads(_repair_json(text))

except ImportError:

    def _loads_robust(text: str) -> dict:  # type: ignore[misc]
        return json.loads(text)


from datetime import datetime
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import nibabel as nib
from scipy.ndimage import binary_erosion

try:
    from PIL import Image
except ImportError:
    Image = None

from search2adapt.agents.base import (
    BaseAgent,
    EvalResult,
    LLMCaller,
    MaskState,
    extract_mask_features,
    features_to_text,
    infer_axial_axis,
)
from search2adapt.config import Search2AdaptConfig
from search2adapt.prompts import PROMPT_REGISTRY

if TYPE_CHECKING:
    from search2adapt.memory import StrategyMemory


# =============================================================================
#  Visualization Utilities (for VLM visual evaluation)
# =============================================================================

# Dark, high-saturation boundary colors per label (from PARDA)
_BOUNDARY_PALETTE = [
    (180, 30, 30),  # deep red
    (30, 160, 60),  # deep green
    (30, 80, 200),  # deep blue
    (140, 60, 170),  # deep purple
    (200, 120, 20),  # deep orange
    (20, 140, 140),  # deep teal
    (120, 120, 120),  # gray
    (180, 20, 120),  # magenta
    (90, 60, 20),  # brown
    (20, 20, 20),  # near black
]


def _normalize_image(image_2d: np.ndarray) -> np.ndarray:
    """1st/99th percentile clipping → 0-255 uint8"""
    img = image_2d.astype(np.float32)
    p_low, p_high = np.percentile(img, [1, 99])
    img = np.clip(img, p_low, p_high)
    denom = img.max() - img.min()
    if denom < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - img.min()) / denom * 255.0
    return img.astype(np.uint8)


def _extract_boundary(mask_2d: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Extract boundary pixels via erosion"""
    eroded = binary_erosion(mask_2d, iterations=thickness)
    return mask_2d & ~eroded


def _create_overlay(
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    opacity: float = 0.2,
    boundary_thickness: int = 1,
) -> "Image.Image":
    """
    Create a multi-class overlay image:
    - Cyan semi-transparent fill for all foreground
    - Per-label colored boundary lines
    """
    if Image is None:
        raise ImportError("Pillow is required for visual evaluation")

    norm = _normalize_image(image_slice)
    rgb = np.stack([norm] * 3, axis=-1).astype(np.float32)

    lbl = label_slice.astype(np.int32)
    max_label = int(lbl.max())
    fill_color = np.array([0, 255, 255], dtype=np.float32)  # cyan

    # Semi-transparent fill
    fg = lbl > 0
    rgb[fg] = (1 - opacity) * rgb[fg] + opacity * fill_color

    # Per-label boundary
    for k in range(1, max_label + 1):
        mk = lbl == k
        if mk.sum() == 0:
            continue
        bd = _extract_boundary(mk, thickness=boundary_thickness)
        color = _BOUNDARY_PALETTE[(k - 1) % len(_BOUNDARY_PALETTE)]
        rgb[bd] = np.array(color, dtype=np.float32)

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _build_image_grid(images: List["Image.Image"], cols: int = 2) -> "Image.Image":
    """
    Compose overlay images into a single grid image.

    Cells are resized to a common size and empty cells are padded with black
    so the grid keeps a rectangular shape for the VLM.
    """
    if Image is None:
        raise ImportError("Pillow is required for visual evaluation")

    if not images:
        return Image.new("RGB", (256, 256), (0, 0, 0))

    cell_w, cell_h = images[0].size
    cells = [img.resize((cell_w, cell_h), Image.LANCZOS) for img in images]

    rows = (len(cells) + cols - 1) // cols
    while len(cells) < rows * cols:
        cells.append(Image.new("RGB", (cell_w, cell_h), (0, 0, 0)))

    grid = Image.new("RGB", (cell_w * cols, cell_h * rows))
    for i, cell in enumerate(cells):
        row, col = divmod(i, cols)
        grid.paste(cell, (col * cell_w, row * cell_h))

    return grid


def _select_mixed_slices(
    label_data: np.ndarray,
    axial_axis: int,
    num_top: int = 2,
    num_random: int = 2,
) -> List[int]:
    """
    Select slices for VLM grid: top-K by label area + random others.
    """
    num_slices = label_data.shape[axial_axis]
    slice_areas = []
    for i in range(num_slices):
        slices = [slice(None)] * 3
        slices[axial_axis] = i
        area = int(np.sum(label_data[tuple(slices)] > 0))
        if area > 0:
            slice_areas.append((i, area))

    if not slice_areas:
        # No foreground — return evenly spaced slices
        step = max(1, num_slices // (num_top + num_random))
        return [i * step for i in range(min(num_top + num_random, num_slices))]

    slice_areas.sort(key=lambda x: x[1], reverse=True)
    top = [s[0] for s in slice_areas[:num_top]]

    remaining = [s for s in slice_areas[num_top:]]
    n_rand = min(num_random, len(remaining))
    rand = [s[0] for s in random.sample(remaining, n_rand)] if n_rand > 0 else []

    return sorted(top + rand)


def _create_grid_image(images: List["Image.Image"]) -> "Image.Image":
    """Create a 2x2 grid from up to 4 images, pad with black if < 4."""
    if Image is None:
        raise ImportError("Pillow is required")

    target_size = images[0].size if images else (256, 256)
    resized = [img.resize(target_size, Image.LANCZOS) for img in images]
    while len(resized) < 4:
        resized.append(Image.new("RGB", target_size, (0, 0, 0)))

    w, h = target_size
    grid = Image.new("RGB", (w * 2, h * 2))
    grid.paste(resized[0], (0, 0))
    grid.paste(resized[1], (w, 0))
    grid.paste(resized[2], (0, h))
    grid.paste(resized[3], (w, h))
    return grid


def _image_to_base64(img: "Image.Image") -> str:
    """PIL Image → base64 string"""
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================================
#  Evaluator Agent
# =============================================================================


class Evaluator(BaseAgent):
    """
    Evaluator (Critic) Agent — MCTS 的 Reward 函数

    两种评估模式:
      1. text_eval: 纯文本 (几何特征 → LLM 打分), 速度快, 每次 MCTS 迭代默认使用
      2. visual_eval: 视觉 (overlay grid → VLM 打分), 更准确, 可选使用

    在 MCTS 中的角色:
      - Phase 3 (Evaluation): 对 Executor 执行后的新状态 s_{t+1} 打分
      - 输出 reward ∈ [0, 1] 用于 backpropagation

    Geometric Hard Constraints:
      _apply_geometric_constraints() applies deterministic penalties to the
      LLM reward based on computed mask statistics:
        • Empty mask (fg_ratio ≈ 0)      → reward = 0.0
        • Severe fragmentation (many CC) → penalty proportional to excess CCs
        • Extreme foreground ratios      → small penalty
      These act as guardrails independent of LLM opinion.
    """

    def __init__(
        self,
        config: Search2AdaptConfig,
        memory: "Optional[StrategyMemory]" = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name="Evaluator", config=config, logger=logger)

        self.llm_caller = LLMCaller(config.llm, self.logger)
        self.vlm_caller = LLMCaller(config.vlm, self.logger)
        self.memory = memory

        self.target_organs: List[str] = []
        self.organ_label_map: Dict[str, int] = {}
        self.modality: str = "Unknown"
        self.anatomical_site: str = "Unknown"

    def set_domain_context(
        self,
        target_organs: List[str],
        organ_label_map: Dict[str, int],
        modality: str = "Unknown",
        anatomical_site: str = "Unknown",
    ):
        """由 Perceptor 分析结果设置领域上下文"""
        self.target_organs = target_organs
        self.organ_label_map = organ_label_map
        self.modality = modality
        self.anatomical_site = anatomical_site

    # ----- Core API -----

    def run(
        self, state: MaskState, mode: str = "text", action_taken: str = ""
    ) -> EvalResult:
        """
        评估 mask 质量, 返回 reward

        Args:
            state: 当前 MaskState
            mode: "text" (LLM), "visual" (VLM), "plan_eval", or "viability"
                  (plan_eval and viability are dispatched via dedicated methods)
            action_taken: tool_id of the action that produced this state
                  (recorded in EvalRecord for action-reward analysis)

        Returns:
            EvalResult with reward, scores, decision, defects
        """
        if mode == "visual":
            return self._visual_eval(state, action_taken=action_taken)
        elif mode == "text":
            return self._text_eval(state, action_taken=action_taken)
        else:
            raise ValueError(
                f"Unsupported evaluation mode: {mode!r}. "
                "Use 'text' or 'visual'. "
                "For plan_eval/viability, call the dedicated methods directly."
            )

    # ----- Text-based Evaluation -----

    def _text_eval(self, state: MaskState, action_taken: str = "") -> EvalResult:
        """
        纯文本评估: 提取 mask 几何特征 → 文本化 → LLM 打分
        """
        # 1. Extract features
        features = extract_mask_features(state.mask_path, self.organ_label_map)
        state.cached_features = features
        features_text = features_to_text(features)

        # 2. Fill prompt template
        prompt = PROMPT_REGISTRY["EVALUATOR_REWARD"].format(
            target_organs=", ".join(self.target_organs),
            mask_features=features_text,
            modality=self.modality,
            anatomical_site=self.anatomical_site,
        )

        # 3. Call LLM
        system_prompt = (
            "You are an expert medical image segmentation quality assessor. "
            "Respond ONLY with valid JSON."
        )
        try:
            response = self.llm_caller.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.1,
            )
            result = self._parse_eval_response(response, mode="text")
        except Exception as e:
            self.logger.error(f"Text evaluation failed: {e}")
            result = self._fallback_eval(features)

        # 4. Apply geometric hard constraints (Critique 2 guardrail)
        result = self._apply_geometric_constraints(result, features)

        self.logger.info(
            f"[TextEval] case={state.case_id} step={state.step} "
            f"reward={result.reward:.3f} decision={result.decision}"
        )

        # 6. Persist eval record to shared memory
        self._persist_eval_record(
            state=state,
            result=result,
            mode="text",
            grid_path="",
            action_taken=action_taken,
        )

        return result

    # ----- Visual Evaluation -----

    def _visual_eval(self, state: MaskState, action_taken: str = "") -> EvalResult:
        """
        视觉评估: 生成 overlay grid → VLM 打分
        """
        if Image is None:
            self.logger.warning("Pillow not available, falling back to text eval")
            return self._text_eval(state, action_taken=action_taken)

        # 1. Extract features (for prompt)
        features = extract_mask_features(state.mask_path, self.organ_label_map)
        state.cached_features = features
        features_text = features_to_text(features)

        # 2. Generate overlay grid
        grid_path = ""
        try:
            grid_path = self._create_eval_grid(state)
        except Exception as e:
            self.logger.warning(f"Grid creation failed: {e}, falling back to text eval")
            return self._text_eval(state, action_taken=action_taken)

        # 3. Fill prompt template
        prompt = PROMPT_REGISTRY["EVALUATOR_VISUAL"].format(
            mask_features=features_text,
        )
        user_prompt = (
            f"Target organs: {', '.join(self.target_organs)}\n\n"
            f"Case: {state.case_id}, Step: {state.step}\n\n"
            f"{prompt}"
        )

        # 4. Call VLM with image
        system_prompt = (
            "You are an expert medical image segmentation quality assessor. "
            "Evaluate the overlay image and respond ONLY with valid JSON."
        )
        try:
            response = self.vlm_caller.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=[grid_path],
                temperature=0.1,
            )
            result = self._parse_eval_response(response, mode="visual")
        except Exception as e:
            self.logger.error(f"Visual evaluation failed: {e}")
            result = self._fallback_eval(features)

        self.logger.info(
            f"[VisualEval] case={state.case_id} step={state.step} "
            f"reward={result.reward:.3f} decision={result.decision}"
        )

        # 4. Apply geometric hard constraints (Critique 2 guardrail)
        result = self._apply_geometric_constraints(result, features)

        # 5. Persist eval record to shared memory
        self._persist_eval_record(
            state=state,
            result=result,
            mode="visual",
            grid_path=grid_path,
            action_taken=action_taken,
        )

        return result

    # ----- Memory Persistence -----

    def _persist_eval_record(
        self,
        state: MaskState,
        result: EvalResult,
        mode: str,
        grid_path: str,
        action_taken: str,
    ):
        """
        Write EvalRecord to shared StrategyMemory (if injected).
        Called at the end of _text_eval() and _visual_eval().
        """
        if self.memory is None:
            return
        # Lazy import to avoid circular dependency at module load time
        from search2adapt.memory import EvalRecord

        record = EvalRecord(
            case_id=state.case_id,
            step=state.step,
            mode=mode,
            timestamp=datetime.now().isoformat(),
            reward=result.reward,
            scores=result.scores,
            decision=result.decision,
            defects=result.suggested_actions,
            reasoning=result.reasoning,
            grid_path=grid_path,
            action_taken=action_taken,
        )
        self.memory.add_eval_record(record)

    # ----- Grid Generation -----

    def _create_eval_grid(self, state: MaskState) -> str:
        """
        Generates overlay grid for VLM evaluation.
        """
        img_obj = nib.load(state.image_path)
        lbl_obj = nib.load(state.mask_path)
        img_data = img_obj.get_fdata()
        lbl_data = np.round(lbl_obj.get_fdata()).astype(np.int16)

        axial_axis = infer_axial_axis(img_data.shape)
        slice_indices = _select_mixed_slices(
            lbl_data, axial_axis, num_top=2, num_random=2
        )

        overlays = []
        for idx in slice_indices:
            slices = [slice(None)] * 3
            slices[axial_axis] = idx
            s = tuple(slices)
            overlay = _create_overlay(img_data[s], lbl_data[s])
            overlays.append(overlay)

        grid_img = _build_image_grid(overlays, cols=2)

        case_dir = os.path.dirname(state.mask_path)
        grid_name = f"eval_{os.path.basename(state.mask_path)}.png"
        grid_path = os.path.join(case_dir, grid_name)
        grid_img.save(grid_path)
        return grid_path

    # ----- Response Parsing -----

    # Score weight definitions — two modes use different dimension names.
    # All dimensions are scored 0.0-1.0; reward = weighted_avg directly → [0, 1]
    _TEXT_SCORE_WEIGHTS: Dict = {
        "anatomical_plausibility": 0.35,
        "topological_integrity": 0.25,
        "boundary_smoothness": 0.20,
        "volumetric_consistency": 0.20,
    }
    _VISUAL_SCORE_WEIGHTS: Dict = {
        "anatomical_plausibility": 0.35,
        "boundary_precision": 0.25,
        "morphological_consistency": 0.20,
        "false_positives": 0.20,
    }
    _PLAN_SCORE_WEIGHTS: Dict = {
        "defect_coverage": 0.30,
        "step_coherence": 0.25,
        "tool_appropriateness": 0.25,
        "expected_effectiveness": 0.20,
    }

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

    def _parse_eval_response(self, response: str, mode: str = "text") -> EvalResult:
        """
        Parse LLM/VLM JSON response into EvalResult.

        Handles both evaluation modes:
          - text mode: anatomical_plausibility, topological_integrity,
                       boundary_smoothness, volumetric_consistency
          - visual mode: anatomical_plausibility, boundary_precision,
                         morphological_consistency, false_positives

        Robust to markdown code fences and malformed JSON.
        """
        clean = response.strip()
        # Remove markdown code fences
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:])
        clean = (
            clean.replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )

        # Extract outermost JSON object using bracket matching
        clean = self._extract_json_object(clean)

        data = _loads_robust(clean)

        scores = data.get("scores", {})
        decision = data.get("decision", "CONTINUE").upper()
        defects = data.get("defects", [])
        reasoning = data.get("reasoning", "")

        # Select the correct weight set based on mode.
        # Auto-detect: if any visual-mode key is present in scores, use visual weights.
        visual_keys = set(self._VISUAL_SCORE_WEIGHTS.keys())
        text_keys = set(self._TEXT_SCORE_WEIGHTS.keys())
        has_visual_keys = bool(
            visual_keys.intersection(scores.keys())
            - text_keys.intersection(scores.keys())
        )

        if mode == "visual" or has_visual_keys:
            weight_map = self._VISUAL_SCORE_WEIGHTS
        else:
            weight_map = self._TEXT_SCORE_WEIGHTS

        # Compute reward deterministically from dimension scores
        weighted_sum = 0.0
        weight_total = 0.0
        for dim, w in weight_map.items():
            val = scores.get(dim)
            if val is not None:
                fval = float(val)
                # Auto-normalize: if any score > 1.0, assume 1-5 integer scale
                # (the prompt asks for 0.0-1.0, but LLMs sometimes return 1-5).
                if fval > 1.0:
                    fval = fval / 5.0
                    scores[dim] = fval  # store normalized value back
                weighted_sum += w * fval
                weight_total += w

        if weight_total > 0:
            weighted_avg = weighted_sum / weight_total
            reward = weighted_avg  # scores are already 0.0-1.0
        else:
            # No dimension scores matched — fall back to LLM-provided reward
            reward = float(data.get("weighted_reward", 0.0))

        # Clamp reward to [0, 1]
        reward = max(0.0, min(1.0, reward))

        # Enforce decision thresholds deterministically (ignore LLM's decision
        # only when it contradicts the computed reward to keep consistency)
        if decision not in ("ACCEPT", "CONTINUE", "REJECT"):
            decision = "CONTINUE"
        # Re-derive decision from reward if LLM decision is inconsistent
        if reward >= 0.85:
            decision = "ACCEPT"
        elif reward < 0.5:
            decision = "REJECT"
        else:
            if decision == "ACCEPT":
                decision = "CONTINUE"  # overrule over-optimistic LLM

        return EvalResult(
            reward=reward,
            scores=scores,
            reasoning=reasoning,
            decision=decision,
            suggested_actions=defects,  # defects serve as action suggestions
        )

    def _fallback_eval(self, features: Dict) -> EvalResult:
        """
        Heuristic fallback when LLM/VLM call fails.
        Uses geometric features to estimate a rough reward.
        """
        per_label = features.get("per_label", {})
        num_labels = features.get("num_labels", 0)
        fg_ratio = features.get("total_foreground_ratio", 0.0)

        if num_labels == 0 or fg_ratio < 1e-6:
            return EvalResult(
                reward=0.0,
                scores={},
                reasoning="Empty mask — no foreground detected",
                decision="REJECT",
                suggested_actions=["Re-generate mask with VFM proposer"],
            )

        # Simple heuristic: penalize too many connected components per label
        topo_penalty = 0.0
        for name, info in per_label.items():
            cc = info.get("connected_components", 1)
            if cc > 3:
                topo_penalty += 0.1 * (cc - 1)

        # Penalize extreme foreground ratios
        vol_penalty = 0.0
        if fg_ratio > 0.5:
            vol_penalty = 0.2
        elif fg_ratio < 0.001:
            vol_penalty = 0.3

        raw = 0.6 - topo_penalty - vol_penalty
        reward = max(0.0, min(1.0, raw))

        decision = "CONTINUE"
        if reward >= 0.85:
            decision = "ACCEPT"
        elif reward < 0.5:
            decision = "REJECT"

        return EvalResult(
            reward=reward,
            scores={"heuristic": reward},
            reasoning="Fallback heuristic evaluation (LLM unavailable)",
            decision=decision,
            suggested_actions=["topology_cleaner"] if topo_penalty > 0 else [],
        )

    # =================================================================
    #  Critique 2: Geometric Hard Constraints
    # =================================================================

    def _apply_geometric_constraints(
        self, result: EvalResult, features: Dict
    ) -> EvalResult:
        """
        Apply deterministic geometric-feature-based penalties to the LLM reward.

        This addresses Reviewer Critique 2: geometric feature extraction should
        act as a *constraint / validation* component, not merely be summarised
        into text for LLM interpretation.

        Penalty rules (applied as multiplicative caps or additive penalties):
          1. Empty mask (fg_ratio ≈ 0)              → reward forced to 0.0
          2. All labels lost (num_labels == 0)       → reward forced to 0.0
          3. Severe fragmentation: any label with
             connected_components > 10               → −0.05 per excess CC
             (capped at −0.30 total)
          4. Hyper-expansion: fg_ratio > 0.60        → −0.15 penalty
          5. Micro-foreground: fg_ratio < 0.0005
             (essentially empty but still non-zero)  → −0.20 penalty

        The modified reward is re-clamped to [0, 1] and the decision threshold
        is re-applied to stay consistent.
        """
        fg_ratio = features.get("total_foreground_ratio", 0.0)
        num_labels = features.get("num_labels", 0)
        per_label = features.get("per_label", {})

        penalty = 0.0
        reasons: List[str] = []

        # --- Rule 1 & 2: completely empty ---
        if num_labels == 0 or fg_ratio < 1e-7:
            new_reward = 0.0
            new_scores = dict(result.scores)
            new_scores["geometric_constraint"] = 0.0
            new_reasoning = (
                result.reasoning
                + " | [GeoConstraint] Empty mask detected — reward zeroed."
            )
            return EvalResult(
                reward=0.0,
                scores=new_scores,
                reasoning=new_reasoning,
                decision="REJECT",
                suggested_actions=result.suggested_actions
                + ["Re-generate initial pseudo-label"],
            )

        # --- Rule 3: severe fragmentation ---
        frag_penalty = 0.0
        for name, info in per_label.items():
            cc = info.get("connected_components", 1)
            if cc > 10:
                excess = cc - 10
                frag_penalty += 0.05 * excess
        frag_penalty = min(frag_penalty, 0.30)
        if frag_penalty > 0:
            penalty += frag_penalty
            reasons.append(f"severe_fragmentation(penalty={frag_penalty:.2f})")

        # --- Rule 4: hyper-expansion ---
        if fg_ratio > 0.60:
            penalty += 0.15
            reasons.append(f"hyper_expansion(fg={fg_ratio:.3f})")

        # --- Rule 5: micro-foreground ---
        if 0 < fg_ratio < 0.0005:
            penalty += 0.20
            reasons.append(f"micro_foreground(fg={fg_ratio:.5f})")

        if penalty <= 0.0:
            return result  # no constraints triggered — return unchanged

        new_reward = max(0.0, min(1.0, result.reward - penalty))
        new_scores = dict(result.scores)
        new_scores["geometric_constraint"] = round(-penalty, 4)

        constraint_note = f" | [GeoConstraint] penalties={reasons} total={penalty:.3f}"
        new_reasoning = result.reasoning + constraint_note

        # Re-derive decision from updated reward
        if new_reward >= 0.85:
            decision = "ACCEPT"
        elif new_reward < 0.5:
            decision = "REJECT"
        else:
            decision = "CONTINUE"

        new_actions = list(result.suggested_actions)
        if "severe_fragmentation" in str(reasons):
            if "topology_cleaner" not in new_actions:
                new_actions.append("topology_cleaner")

        return EvalResult(
            reward=new_reward,
            scores=new_scores,
            reasoning=new_reasoning,
            decision=decision,
            suggested_actions=new_actions,
        )

    # =================================================================
    #  CoTS Phase 1: Plan-level Evaluation (Critic role)
    # =================================================================

    def plan_eval(
        self,
        state: MaskState,
        proposed_plan: Dict,
        eval_history: str = "",
        iteration: int = 0,
    ) -> Tuple[float, List[str], str]:
        """
        Evaluate a proposed refinement plan's quality via LLM — CoTS Critic role.

        This is called during Phase 1 (plan-level tree search).  No tool
        execution happens; the Evaluator scores the *plan text* across four
        dimensions using the EVALUATOR_PLAN_CRITIQUE prompt.

        Args:
            state:         Current MaskState (used for context in prompt).
            proposed_plan: Dict with keys ``steps``, ``rationale``,
                           ``target_defects``, ``expected_outcome`` — typically
                           ``RefinementPlan.to_dict()``.
            eval_history:  Formatted text of prior eval records for this case
                           (from memory.get_case_search_context). Injected into
                           the prompt so the Critic can avoid re-scoring already-
                           tried approaches.
            iteration:     Current plan search iteration number (for logging /
                           persisting the EvalRecord step field).

        Returns:
            (plan_reward, suggestions, critique)
              - plan_reward: float in [0, 1], weighted score
              - suggestions: list of improvement suggestions from the Critic
              - critique: full critique text from the Critic
        """
        # 1. Build state description
        state_desc = self._build_state_description_for_plan(state)

        # 2. Domain profile summary from memory
        if not self.config.mcts.enable_perceptor_context:
            domain_profile_summary = "(Domain profile DISABLED for ablation)"
        else:
            domain_profile_summary = (
                self.memory.get_domain_context_summary()
                if self.memory is not None
                else "(Domain profile not available)"
            )

        # 3. Fallback case eval history
        if not eval_history:
            eval_history = (
                self.memory.get_case_search_context(state.case_id)
                if self.memory is not None
                else "(No evaluation history available)"
            )

        # 4. Serialize proposed plan for prompt
        plan_text = json.dumps(proposed_plan, indent=2, ensure_ascii=False)

        # 5. Fill prompt template (new fields: domain_profile_summary, eval_history)
        prompt = PROMPT_REGISTRY["EVALUATOR_PLAN_CRITIQUE"].format(
            domain_profile_summary=domain_profile_summary,
            eval_history=eval_history,
            state_description=state_desc,
            target_organs=", ".join(self.target_organs),
            modality=self.modality,
            anatomical_site=self.anatomical_site,
            proposed_plan=plan_text,
        )

        system_prompt = (
            "You are an expert medical image processing Critic in a Collaborative "
            "Tree Search framework. Evaluate the proposed plan and respond ONLY "
            "with valid JSON."
        )

        plan_reward: float
        suggestions: List[str]
        critique: str
        scores: Dict = {}

        try:
            response = self.llm_caller.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.1,
                no_think=True,
            )
            plan_reward, suggestions, critique, scores = self._parse_plan_eval_response(
                response
            )
        except Exception as e:
            self.logger.error(f"Plan evaluation failed: {e}")
            plan_reward = 0.5
            suggestions = []
            critique = f"Plan evaluation failed: {e}"

        self.logger.info(
            f"[PlanEval] case={state.case_id} iter={iteration} "
            f"plan_reward={plan_reward:.3f} "
            f"suggestions={len(suggestions)}"
        )
        if critique:
            self.logger.debug(f"[PlanEval] reasoning: {critique[:200]}")

        # 6. Persist plan eval record to memory
        if self.memory is not None:
            plan_tools = " → ".join(
                s.get("tool_id", "?") for s in proposed_plan.get("steps", [])
            )
            self.memory.add_plan_eval_record(
                case_id=state.case_id,
                step=iteration,
                mode="plan_eval",
                reward=plan_reward,
                decision="ACCEPT"
                if plan_reward >= 0.85
                else ("REJECT" if plan_reward < 0.5 else "CONTINUE"),
                defects=suggestions,
                reasoning=critique,
                action_taken=plan_tools,
                scores=scores,
            )

        return plan_reward, suggestions, critique

    def viability_eval(
        self,
        original_plan: Dict,
        executed_steps: List[Dict],
        current_state: MaskState,
        remaining_steps: List[Dict],
        case_eval_history: str = "",
    ) -> Tuple[float, bool]:
        """
        Mid-execution plan viability check — CoTS Phase 2.

        Called after each step during Phase 2 plan execution.  Assesses
        whether the remaining plan steps are still appropriate given the
        intermediate mask quality.

        Args:
            original_plan:     Full plan dict (``RefinementPlan.to_dict()``).
            executed_steps:    Steps already executed (list of step dicts).
            current_state:     MaskState after executed steps.
            remaining_steps:   Steps not yet executed (list of step dicts).
            case_eval_history: Formatted eval history for this case (from memory).
                               Injected into the prompt so the monitor has context
                               about what prior attempts produced for this case.

        Returns:
            (viability_reward, still_viable)
              - viability_reward: float in [0, 1]
              - still_viable: True if viability_reward >= viability_threshold
        """
        viability_threshold = self.config.mcts.viability_threshold

        # Build state description with mask features
        state_desc = self._build_state_description_for_plan(current_state)

        # Domain profile summary from memory
        domain_profile_summary = (
            self.memory.get_domain_context_summary()
            if self.memory is not None
            else "(Domain profile not available)"
        )

        # Fallback case eval history
        if not case_eval_history:
            case_eval_history = (
                self.memory.get_case_search_context(current_state.case_id)
                if self.memory is not None
                else "(No evaluation history available)"
            )

        prompt = PROMPT_REGISTRY["EVALUATOR_PLAN_VIABILITY"].format(
            domain_profile_summary=domain_profile_summary,
            case_eval_history=case_eval_history,
            original_plan=json.dumps(original_plan, indent=2, ensure_ascii=False),
            executed_steps=json.dumps(executed_steps, indent=2, ensure_ascii=False),
            current_state=state_desc,
            remaining_steps=json.dumps(remaining_steps, indent=2, ensure_ascii=False),
        )

        system_prompt = (
            "You are an expert plan viability monitor for medical image "
            "segmentation. Assess whether remaining plan steps are still "
            "appropriate. Respond ONLY with valid JSON."
        )

        viability_reward: float
        still_viable: bool
        response: str = ""

        try:
            response = self.llm_caller.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.1,
                no_think=True,
            )
            viability_reward, still_viable = self._parse_viability_response(response)
        except Exception as e:
            self.logger.error(f"Viability check failed: {e}")
            # On failure, assume plan is still viable to avoid premature abort
            viability_reward = 0.6
            still_viable = True

        # Override still_viable with threshold check
        still_viable = viability_reward >= viability_threshold

        self.logger.info(
            f"[ViabilityEval] case={current_state.case_id} "
            f"viability={viability_reward:.3f} "
            f"still_viable={still_viable}"
        )

        # Log reasoning from viability response
        try:
            _clean = response.strip()
            _clean = self._extract_json_object(
                _clean.replace("```json", "").replace("```", "").strip()
            )
            _data = _loads_robust(_clean)
            _reasoning = _data.get("reasoning", "")
            if _reasoning:
                self.logger.debug(f"[ViabilityEval] reasoning: {_reasoning[:200]}")
        except Exception:
            pass

        # Persist viability eval record to memory
        if self.memory is not None:
            executed_tools = " → ".join(s.get("tool_id", "?") for s in executed_steps)
            self.memory.add_plan_eval_record(
                case_id=current_state.case_id,
                step=len(executed_steps),
                mode="viability",
                reward=viability_reward,
                decision="CONTINUE" if still_viable else "REJECT",
                defects=[],
                reasoning=f"Viability check after {len(executed_steps)} steps: "
                f"still_viable={still_viable}",
                action_taken=executed_tools,
                scores={},
            )

        return viability_reward, still_viable

    # ----- Plan Eval Response Parsing -----

    def _parse_plan_eval_response(
        self, response: str
    ) -> Tuple[float, List[str], str, Dict]:
        """
        Parse LLM JSON response for plan evaluation (EVALUATOR_PLAN_CRITIQUE).

        Returns:
            (plan_reward, suggestions, critique, scores)
        """
        clean = response.strip()
        # Remove markdown code fences
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
        data = _loads_robust(clean)

        scores = data.get("scores", {})
        suggestions = data.get("suggestions", [])
        # Prompt now outputs "reasoning"; fall back to legacy "critique" field
        critique = data.get("reasoning", "") or data.get("critique", "")

        # Compute plan_reward from dimension scores using plan weights
        weight_map = self._PLAN_SCORE_WEIGHTS
        weighted_sum = 0.0
        weight_total = 0.0
        for dim, w in weight_map.items():
            val = scores.get(dim)
            if val is not None:
                fval = float(val)
                # Auto-normalize: if score > 1.0, assume 1-5 integer scale
                if fval > 1.0:
                    fval = fval / 5.0
                    scores[dim] = fval
                weighted_sum += w * fval
                weight_total += w

        if weight_total > 0:
            weighted_avg = weighted_sum / weight_total
            plan_reward = weighted_avg  # scores are already 0.0-1.0
        else:
            plan_reward = float(data.get("weighted_reward", 0.5))

        plan_reward = max(0.0, min(1.0, plan_reward))

        return plan_reward, suggestions, critique, scores

    def _parse_viability_response(self, response: str) -> Tuple[float, bool]:
        """
        Parse LLM JSON response for viability check (EVALUATOR_PLAN_VIABILITY).

        Returns:
            (viability_reward, still_viable)
        """
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
        data = _loads_robust(clean)

        viability_reward = float(data.get("viability_reward", 0.5))
        viability_reward = max(0.0, min(1.0, viability_reward))

        still_viable = bool(data.get("still_viable", True))

        return viability_reward, still_viable

    # ----- Plan eval helpers -----

    def _build_state_description_for_plan(self, state: MaskState) -> str:
        """Build a text description of the current mask state for plan prompts."""
        lines = [
            f"Case: {state.case_id}",
            f"Step: {state.step}",
            f"Target organs: {', '.join(self.target_organs)}",
            f"Modality: {self.modality}",
            f"Anatomical site: {self.anatomical_site}",
        ]

        if state.mask_path and state.mask_path != "":
            try:
                features = extract_mask_features(state.mask_path, self.organ_label_map)
                state.cached_features = features
                lines.append("")
                lines.append("Current mask geometric features:")
                lines.append(features_to_text(features))
            except Exception as e:
                lines.append(f"(Could not extract features: {e})")
        else:
            lines.append("No mask available yet (initial state).")

        return "\n".join(lines)

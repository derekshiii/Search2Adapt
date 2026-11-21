"""
Search2Adapt Reward Credibility Analysis Module (Critique 1)

Addresses Reviewer Critique 1: "LLM/VLM-based rewards may not correlate
with real segmentation quality."

This module computes:
  - Pearson and Spearman correlation between LLM rewards and GT Dice scores
  - Per-dimension correlation analysis (which scoring dimension best predicts
    true quality?)
  - Reward stability: variance across repeated evaluations of the same mask
  - Summary report saved as reward_credibility.json

Usage::

    from search2adapt.analysis import reward_credibility_report
    report = reward_credibility_report(
        eval_records=memory.get_all_eval_records(),
        gt_metrics=gt_metrics_dict,      # from metrics.evaluate_result_dir()
        result_dir=config.result_dir,
    )

The module is self-contained and has no dependency on agent classes.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Correlation helpers
# ---------------------------------------------------------------------------


def _pearson(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient and two-tailed p-value.

    Returns (r, p_value). Falls back to (nan, nan) for degenerate input.
    """
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    n = len(x_arr)
    if n < 3:
        return float("nan"), float("nan")

    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(x_arr, y_arr)
        return float(r), float(p)
    except Exception:
        pass

    # Pure numpy fallback
    xm = x_arr - x_arr.mean()
    ym = y_arr - y_arr.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom < 1e-12:
        return float("nan"), float("nan")
    r = float((xm * ym).sum() / denom)
    # Approximate p-value via t-distribution with n-2 df
    try:
        t_stat = r * np.sqrt((n - 2) / max(1e-12, 1 - r**2))
        from scipy.stats import t as t_dist

        p = float(2 * t_dist.sf(abs(t_stat), df=n - 2))
    except Exception:
        p = float("nan")
    return r, p


def _spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation coefficient and two-tailed p-value.
    """
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if len(x_arr) < 3:
        return float("nan"), float("nan")

    try:
        from scipy.stats import spearmanr

        r, p = spearmanr(x_arr, y_arr)
        return float(r), float(p)
    except Exception:
        pass

    # Pure numpy fallback: rank then pearson
    def _rank(a):
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float) + 1
        return ranks

    rx = _rank(x_arr)
    ry = _rank(y_arr)
    return _pearson(list(rx), list(ry))


def _nan_safe_json(obj: Any) -> Any:
    """Recursively replace nan/inf with None for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _nan_safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_safe_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
#  Main analysis entry-point
# ---------------------------------------------------------------------------


def reward_credibility_report(
    eval_records: List[Dict],
    gt_metrics: Dict,
    result_dir: Optional[str] = None,
    execution_modes: Optional[List[str]] = None,
) -> Dict:
    """
    Compute a reward credibility report comparing LLM-based rewards against
    ground-truth Dice scores.

    Args:
        eval_records:    List of EvalRecord dicts (from memory or JSON).
                         Each dict must have at minimum:
                           {"case_id", "reward", "scores", "mode", "step"}
        gt_metrics:      Output of metrics.evaluate_result_dir() or a dict
                         of {case_id: {"mean_dice": float, "per_organ": {...}}}.
                         Accepts both formats.
        result_dir:      If given, saves report as reward_credibility.json here.
        execution_modes: Filter eval records by mode (e.g. ["text", "visual"]).
                         Default: include all modes except "plan_eval".

    Returns:
        {
          "n_pairs":              int,   (number of (reward, dice) pairs)
          "pearson_r":            float,
          "pearson_p":            float,
          "spearman_r":           float,
          "spearman_p":           float,
          "mean_llm_reward":      float,
          "std_llm_reward":       float,
          "mean_gt_dice":         float,
          "std_gt_dice":          float,
          "reward_stability":     {case_id: {"std": float, "n": int}, ...},
          "per_dimension_corr":   {dim: {"pearson_r": float, "spearman_r": float}, ...},
          "scatter_data":         [{"case_id", "reward", "dice", "mode", "step"}, ...],
          "interpretation":       str,
        }
    """
    if execution_modes is None:
        execution_modes = ["text", "visual", "fallback"]

    # -------------------------------------------------------------------
    # 1. Build lookup: case_id → GT mean_dice
    #    Support both nested {"per_case": {case_id: {"mean_dice": ...}}}
    #    and flat {case_id: {"mean_dice": ...}} formats.
    # -------------------------------------------------------------------
    gt_lookup: Dict[str, float] = {}

    per_case = gt_metrics.get("per_case", None)
    if per_case is not None:
        # evaluate_result_dir() format
        for cid, cm in per_case.items():
            if isinstance(cm, dict) and "mean_dice" in cm:
                d = cm["mean_dice"]
                if d is not None and not (isinstance(d, float) and np.isnan(d)):
                    gt_lookup[cid] = float(d)
    else:
        # Flat format: {case_id: {"mean_dice": ...}}
        for cid, cm in gt_metrics.items():
            if isinstance(cm, dict) and "mean_dice" in cm:
                d = cm["mean_dice"]
                if d is not None and not (isinstance(d, float) and np.isnan(d)):
                    gt_lookup[cid] = float(d)

    if not gt_lookup:
        logger.warning(
            "No GT Dice values found in gt_metrics — cannot compute correlation"
        )
        return {"error": "no_gt_dice_values", "n_pairs": 0}

    # -------------------------------------------------------------------
    # 2. Filter eval records and pair with GT Dice
    # -------------------------------------------------------------------
    scatter: List[Dict] = []
    per_case_rewards: Dict[str, List[float]] = {}

    for rec in eval_records:
        mode = rec.get("mode", "")
        if mode not in execution_modes:
            continue

        case_id = rec.get("case_id", "")
        reward = rec.get("reward")
        if reward is None or (isinstance(reward, float) and np.isnan(reward)):
            continue

        # Match case_id to GT: try exact and prefix match
        dice = gt_lookup.get(case_id)
        if dice is None:
            # Try stripping trailing suffixes added by pipeline
            base = case_id.split("_step")[0]
            dice = gt_lookup.get(base)
        if dice is None:
            continue  # no GT for this case

        scatter.append(
            {
                "case_id": case_id,
                "reward": float(reward),
                "dice": float(dice),
                "mode": mode,
                "step": rec.get("step", -1),
            }
        )
        per_case_rewards.setdefault(case_id, []).append(float(reward))

    if not scatter:
        logger.warning(
            "No eval records matched GT cases after filtering by mode. "
            f"Available modes in records: "
            f"{set(r.get('mode', '') for r in eval_records)}, "
            f"GT case IDs: {list(gt_lookup.keys())}"
        )
        return {"error": "no_matching_pairs", "n_pairs": 0}

    rewards = [s["reward"] for s in scatter]
    dices = [s["dice"] for s in scatter]

    # -------------------------------------------------------------------
    # 3. Pearson / Spearman
    # -------------------------------------------------------------------
    pearson_r, pearson_p = _pearson(rewards, dices)
    spearman_r, spearman_p = _spearman(rewards, dices)

    # -------------------------------------------------------------------
    # 4. Reward stability per case (std of repeated evals)
    # -------------------------------------------------------------------
    stability: Dict[str, Dict] = {}
    for cid, rews in per_case_rewards.items():
        stability[cid] = {
            "std": float(np.std(rews)) if len(rews) > 1 else float("nan"),
            "mean": float(np.mean(rews)),
            "n": len(rews),
        }

    # -------------------------------------------------------------------
    # 5. Per-dimension correlation with GT Dice
    # -------------------------------------------------------------------
    dim_corr: Dict[str, Dict] = {}

    # Collect dimension → (score_vals, dice_vals) where both known
    dim_pairs: Dict[str, Tuple[List[float], List[float]]] = {}
    for rec in eval_records:
        mode = rec.get("mode", "")
        if mode not in execution_modes:
            continue
        case_id = rec.get("case_id", "")
        dice = gt_lookup.get(case_id)
        if dice is None:
            base = case_id.split("_step")[0]
            dice = gt_lookup.get(base)
        if dice is None:
            continue
        scores = rec.get("scores", {})
        for dim, val in scores.items():
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if dim not in dim_pairs:
                dim_pairs[dim] = ([], [])
            dim_pairs[dim][0].append(fval)
            dim_pairs[dim][1].append(float(dice))

    for dim, (sv, dv) in dim_pairs.items():
        if len(sv) < 3:
            continue
        pr, pp = _pearson(sv, dv)
        sr, sp = _spearman(sv, dv)
        dim_corr[dim] = {
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "n": len(sv),
        }

    # -------------------------------------------------------------------
    # 6. Interpretation string
    # -------------------------------------------------------------------
    if not np.isnan(spearman_r):
        if spearman_r >= 0.7:
            interp = (
                f"Strong positive correlation (ρ={spearman_r:.3f}) between "
                "LLM reward and GT Dice — reward is credible."
            )
        elif spearman_r >= 0.4:
            interp = (
                f"Moderate positive correlation (ρ={spearman_r:.3f}). "
                "LLM reward partially tracks real quality."
            )
        elif spearman_r >= 0.0:
            interp = (
                f"Weak positive correlation (ρ={spearman_r:.3f}). "
                "LLM reward has limited reliability; consider hybrid reward."
            )
        else:
            interp = (
                f"Negative / no correlation (ρ={spearman_r:.3f}). "
                "LLM reward does NOT track GT quality — use hybrid reward."
            )
    else:
        interp = "Insufficient data to compute correlation."

    report = {
        "n_pairs": len(scatter),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "mean_llm_reward": float(np.mean(rewards)),
        "std_llm_reward": float(np.std(rewards)),
        "mean_gt_dice": float(np.mean(dices)),
        "std_gt_dice": float(np.std(dices)),
        "reward_stability": stability,
        "per_dimension_corr": dim_corr,
        "scatter_data": scatter,
        "interpretation": interp,
    }

    if result_dir:
        out_path = os.path.join(result_dir, "reward_credibility.json")
        with open(out_path, "w") as f:
            json.dump(_nan_safe_json(report), f, indent=2)
        logger.info(f"Reward credibility report saved to {out_path}")

    logger.info(
        f"[CredibilityReport] n={len(scatter)} "
        f"pearson_r={pearson_r:.3f} spearman_r={spearman_r:.3f} | {interp}"
    )

    return report


# ---------------------------------------------------------------------------
#  Step-level delta analysis  (does each tool actually improve the mask?)
# ---------------------------------------------------------------------------


def step_improvement_analysis(
    eval_records: List[Dict],
    gt_step_metrics: Dict[str, List[Dict]],
) -> Dict:
    """
    Analyse whether LLM step rewards track per-step GT Dice improvements.

    Args:
        eval_records:    Execution eval records (mode="text" or "visual").
        gt_step_metrics: {case_id: [{"step": int, "tool": str, "dice": float}, ...]}
                         Per-step GT Dice from e.g. a step-by-step evaluation run.

    Returns:
        {
          "llm_delta_vs_gt_delta": [{"case_id", "step", "llm_delta", "gt_delta"}, ...],
          "pearson_r":  float,
          "spearman_r": float,
        }
    """
    # Build lookup: (case_id, step) → reward
    reward_lookup: Dict[Tuple[str, int], float] = {}
    for rec in eval_records:
        mode = rec.get("mode", "")
        if mode not in ("text", "visual"):
            continue
        cid = rec.get("case_id", "")
        step = int(rec.get("step", -1))
        rew = rec.get("reward")
        if rew is not None:
            reward_lookup[(cid, step)] = float(rew)

    pairs: List[Dict] = []
    for case_id, steps in gt_step_metrics.items():
        sorted_steps = sorted(steps, key=lambda s: s["step"])
        for i in range(1, len(sorted_steps)):
            prev = sorted_steps[i - 1]
            curr = sorted_steps[i]
            gt_delta = curr["dice"] - prev["dice"]
            rew_curr = reward_lookup.get((case_id, curr["step"]))
            rew_prev = reward_lookup.get((case_id, prev["step"]))
            if rew_curr is None or rew_prev is None:
                continue
            llm_delta = rew_curr - rew_prev
            pairs.append(
                {
                    "case_id": case_id,
                    "step": curr["step"],
                    "tool": curr.get("tool", "?"),
                    "llm_delta": llm_delta,
                    "gt_delta": gt_delta,
                }
            )

    if len(pairs) < 3:
        return {"error": "insufficient_data", "llm_delta_vs_gt_delta": pairs}

    llm_deltas = [p["llm_delta"] for p in pairs]
    gt_deltas = [p["gt_delta"] for p in pairs]
    pr, _ = _pearson(llm_deltas, gt_deltas)
    sr, _ = _spearman(llm_deltas, gt_deltas)

    return {
        "llm_delta_vs_gt_delta": pairs,
        "pearson_r": pr,
        "spearman_r": sr,
        "n_deltas": len(pairs),
    }

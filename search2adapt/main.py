#!/usr/bin/env python
"""
Search2Adapt — Multi-Agent Collaborative Tree Search for SFUDA

Main entry point. Orchestrates the full pipeline:
  Phase 1:  Perceptor     → Domain analysis (domain_profile → unified memory)
            Strategist    → Adaptation plan generation (adaptation_plans → unified memory)
  Phase 2:  MCTS Search   → Per-case pseudo-label optimization (trajectories → unified memory)
  Phase 3:  Distillation  → nnUNet dataset construction + training

Shared Memory Design:
  A single StrategyMemory instance is created here and passed to all agents.
  All three layers (domain_profile / adaptation_plans / trajectories) are
  written to ONE file: memory_<domain_key>.json
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from search2adapt.ablation_configs import apply_ablation_config
from search2adapt.agents.perceptor import Perceptor
from search2adapt.agents.strategist import Strategist
from search2adapt.config import (
    MCTSConfig,
    PerceptorConfig,
    Search2AdaptConfig,
)
from search2adapt.memory import StrategyMemory
from search2adapt.utils import print_config_summary, setup_logger

# ================================================================
#  Result dir naming: {dataset_name}_{YYYYMMDD_HHMMSS}
# ================================================================


def _make_result_dir(
    base_dir: str, dataset_path: str, ablation_name: str = "full"
) -> str:
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ablation_name and ablation_name != "full":
        return os.path.join(base_dir, f"{dataset_name}_{ablation_name}_{ts}")
    return os.path.join(base_dir, f"{dataset_name}_{ts}")


def _make_shared_memory(
    config: Search2AdaptConfig, logger: logging.Logger
) -> StrategyMemory:
    """Create and return the shared StrategyMemory instance."""
    domain_key = os.path.basename(os.path.normpath(config.target_data_path))
    return StrategyMemory(
        memory_dir=os.path.join(config.result_dir, "memory"),
        domain_key=domain_key,
        max_trajectories=config.memory.max_trajectories,
        prior_top_k=config.memory.prior_top_k,
        prior_min_reward=config.memory.prior_min_reward,
        prior_accept_threshold=config.memory.prior_accept_threshold,
        logger=logger,
    )


# ================================================================
#  Phase 1: run Perceptor + Strategist (joint test / standalone)
# ================================================================


def run_phase1(
    target_data_path: str,
    target_organs: List[str],
    result_base_dir: str = "./S2A_Results",
    grid_size: int = 3,
    num_cases: int = 5,
    cleanup_temp: bool = False,
    num_plans: int = 4,
) -> Dict:
    """
    Run Phase 1: Perceptor (domain analysis) + Strategist (plan generation).

    Both agents share a single StrategyMemory instance, writing to one unified
    memory_<domain>.json file.

    Args:
        target_data_path: nnUNet dataset path (contains imagesTr/)
        target_organs:    list of organ name strings
        result_base_dir:  parent directory for results
        grid_size:        N for N×N grid images sent to VLM
        num_cases:        how many random cases to sample from the dataset
        cleanup_temp:     whether to delete temp grid images after analysis
        num_plans:        number of adaptation plans for Strategist to generate

    Returns:
        {
          "adaptation_plan":    Perceptor output dict,
          "plans":              List[AdaptationPlan],
          "memory_path":        path to unified memory JSON,
        }
    """
    result_dir = _make_result_dir(result_base_dir, target_data_path)

    config = Search2AdaptConfig(
        experiment_name="phase1_test",
        result_dir=result_dir,
        target_organs=",".join(target_organs),
        target_data_path=target_data_path,
        perceptor=PerceptorConfig(
            grid_size=grid_size,
            num_cases=num_cases,
            slices_per_case=grid_size * grid_size,
            cleanup_temp=cleanup_temp,
        ),
    )

    os.makedirs(config.result_dir, exist_ok=True)
    logger = setup_logger(
        name="Search2Adapt",
        log_dir=config.result_dir,
        level="INFO",
    )

    logger.info(f"Result dir: {config.result_dir}")
    logger.info(f"Dataset:    {target_data_path}")
    logger.info(f"Targets:    {target_organs}")
    logger.info(f"Grid:       {grid_size}x{grid_size}")
    logger.info(f"Cases:      {num_cases}")
    logger.info(f"Num plans:  {num_plans}")

    # --- Shared memory (created FIRST, passed to both agents) ---
    memory = _make_shared_memory(config, logger)
    logger.info(f"Shared memory: {memory.unified_file}")

    # --- Perceptor ---
    logger.info("\n" + "#" * 60)
    logger.info("PHASE 1a: Perceptor — Domain Analysis")
    logger.info("#" * 60)

    perceptor = Perceptor(config, memory=memory, logger=logger)
    adaptation_plan = perceptor.run(target_organs)

    logger.info(f"\n{'=' * 60}")
    logger.info("Perceptor complete!")
    logger.info(f"  Targets:  {list(adaptation_plan.get('text_prompts', {}).keys())}")
    logger.info(f"  Modality: {adaptation_plan['domain_context'].get('modality', '?')}")
    logger.info(
        f"  Site:     {adaptation_plan['domain_context'].get('anatomical_site', '?')}"
    )
    logger.info(
        f"  Contrast: {adaptation_plan['domain_context'].get('contrast_quality', '?')}/10"
    )
    logger.info(
        f"  Noise:    {adaptation_plan['domain_context'].get('noise_assessment', '?')}"
    )
    ip = adaptation_plan.get("image_profile", {})
    if ip:
        if "likely_ct" in ip:
            logger.info(f"  Likely CT:      {ip['likely_ct']}")
        if "noise_level_auto" in ip:
            logger.info(f"  Noise (auto):   {ip['noise_level_auto']}")
        if "snr_estimate" in ip:
            logger.info(f"  SNR (auto):     {ip['snr_estimate'].get('mean', '?')}")
    logger.info(f"  Memory:   {adaptation_plan.get('memory_path', '?')}")
    logger.info(f"{'=' * 60}")

    # --- Strategist ---
    logger.info("\n" + "#" * 60)
    logger.info("PHASE 1b: Strategist — Adaptation Plan Generation")
    logger.info("#" * 60)

    strategist = Strategist(config, memory=memory, logger=logger)
    strategist.set_domain_context(
        target_organs=list(adaptation_plan.get("text_prompts", {}).keys()),
        organ_label_map=adaptation_plan["organ_label_map"],
    )

    plans = strategist.generate_adaptation_plans(
        perceptor_plan=adaptation_plan,
        num_plans=num_plans,
        force_regenerate=False,
    )

    logger.info(f"\n{'=' * 60}")
    logger.info("Strategist complete!")
    logger.info(f"  Generated {len(plans)} adaptation plans:")
    for p in plans:
        steps_str = " → ".join(s["tool_id"] for s in p.steps)
        logger.info(
            f"    [{p.priority}] {p.plan_id}: {p.name} | "
            f"steps=[{steps_str}] | "
            f"expected_reward={p.expected_reward:.2f} | "
            f"source={p.source}"
        )
    logger.info(f"  Memory:   {memory.unified_file}")
    logger.info(f"{'=' * 60}")

    return {
        "adaptation_plan": adaptation_plan,
        "plans": plans,
        "memory_path": memory.unified_file,
    }


# ================================================================
#  Phase 1 only: run Perceptor for testing (kept for compatibility)
# ================================================================


def run_perceptor_only(
    target_data_path: str,
    target_organs: List[str],
    result_base_dir: str = "./S2A_Results",
    grid_size: int = 3,
    num_cases: int = 5,
    cleanup_temp: bool = False,
) -> Dict:
    """
    Run only the Perceptor (Phase 1a) for domain analysis testing.
    Uses shared StrategyMemory (writes to unified JSON).

    Returns:
        adaptation_plan dict (from Perceptor.run())
    """
    result_dir = _make_result_dir(result_base_dir, target_data_path)

    config = Search2AdaptConfig(
        experiment_name="perceptor_test",
        result_dir=result_dir,
        target_organs=",".join(target_organs),
        target_data_path=target_data_path,
        perceptor=PerceptorConfig(
            grid_size=grid_size,
            num_cases=num_cases,
            slices_per_case=grid_size * grid_size,
            cleanup_temp=cleanup_temp,
        ),
    )

    os.makedirs(config.result_dir, exist_ok=True)
    logger = setup_logger(
        name="Search2Adapt",
        log_dir=config.result_dir,
        level="INFO",
    )

    logger.info(f"Result dir: {config.result_dir}")
    logger.info(f"Dataset:    {target_data_path}")
    logger.info(f"Targets:    {target_organs}")
    logger.info(f"Grid:       {grid_size}x{grid_size}")
    logger.info(f"Cases:      {num_cases}")

    memory = _make_shared_memory(config, logger)

    perceptor = Perceptor(config, memory=memory, logger=logger)
    plan = perceptor.run(target_organs)

    logger.info(f"\n{'=' * 60}")
    logger.info("Perceptor test complete!")
    logger.info(f"  Targets: {list(plan.get('text_prompts', {}).keys())}")
    logger.info(f"  Modality: {plan['domain_context'].get('modality', '?')}")
    logger.info(f"  Site: {plan['domain_context'].get('anatomical_site', '?')}")
    logger.info(f"  Contrast: {plan['domain_context'].get('contrast_quality', '?')}/10")
    logger.info(f"  Noise: {plan['domain_context'].get('noise_assessment', '?')}")
    logger.info(f"  Artifacts: {plan['domain_context'].get('artifacts', '?')}")
    ip = plan.get("image_profile", {})
    if ip:
        logger.info("  [Image Profile]")
        if "likely_ct" in ip:
            logger.info(f"    Likely CT: {ip['likely_ct']}")
        if "noise_level_auto" in ip:
            logger.info(f"    Noise (auto): {ip['noise_level_auto']}")
        if "snr_estimate" in ip:
            logger.info(f"    SNR (auto): {ip['snr_estimate'].get('mean', '?')}")
        if "foreground_ratio" in ip:
            logger.info(
                f"    Foreground ratio: {ip['foreground_ratio'].get('mean', '?'):.2%}"
            )
        if "histogram_num_peaks" in ip:
            logger.info(
                f"    Histogram peaks: {ip['histogram_num_peaks'].get('mean', '?')}"
            )
    logger.info(f"  Plan saved to: {plan.get('memory_path', '?')}")
    logger.info(f"{'=' * 60}")

    return plan


# ================================================================
#  Full pipeline (Phase 1 + 2 + 3) — to be called when ready
# ================================================================


def run_full_pipeline(
    target_data_path: str,
    target_organs: List[str],
    result_base_dir: str = "./S2A_Results",
    grid_size: int = 3,
    num_cases: int = 5,
    mcts_num_cases: int = -1,
    cleanup_temp: bool = False,
    initial_method: str = "biomedparse",
    skip_distillation: bool = False,
    min_reward: float = 0.0,
    num_plans: int = 4,
    mcts_overrides: Optional[Dict] = None,
    ablation_name: str = "full",
) -> Dict:
    """
    Run the full Search2Adapt pipeline (Perceptor + Strategist + MCTS + Distillation).

    A single StrategyMemory is created here and shared by all agents.

    Args:
        target_data_path:  nnUNet 格式数据集路径 (包含 imagesTr/ 目录)
        target_organs:     目标器官列表 (用于 VLM text prompts)
        result_base_dir:   结果输出父目录
        grid_size:         Phase 1 域分析时 VLM 网格图像的 N×N 大小
        num_cases:         Phase 1 Perceptor 域分析的采样 case 数量
        mcts_num_cases:    Phase 2 MCTS 搜索处理的 case 数量; -1 表示处理全部
        cleanup_temp:      是否删除临时图片
        initial_method:    初始伪标签生成方法 ("biomedparse", "sam3", "dual_expert")
        skip_distillation: 是否跳过 Phase 3 知识蒸馏
        min_reward:        Phase 3 蒸馏时，伪标签的最低 reward 阈值
        num_plans:         Strategist 生成的 adaptation plans 数量
        mcts_overrides:    MCTS 超参数覆盖，如 {"plan_max_iterations": 3, "top_k_execute": 1}
    """
    from search2adapt.distillation import DatasetBuilder, DistillationRunner
    from search2adapt.mcts.search import run_batch_search

    result_dir = _make_result_dir(result_base_dir, target_data_path, ablation_name)

    mcts_cfg = MCTSConfig(**(mcts_overrides or {}))

    config = Search2AdaptConfig(
        experiment_name="search2adapt_run",
        result_dir=result_dir,
        target_organs=",".join(target_organs),
        target_data_path=target_data_path,
        perceptor=PerceptorConfig(
            grid_size=grid_size,
            num_cases=num_cases,
            slices_per_case=grid_size * grid_size,
            cleanup_temp=cleanup_temp,
        ),
        mcts=mcts_cfg,
    )

    if ablation_name and ablation_name != "full":
        config = apply_ablation_config(config, ablation_name)

    os.makedirs(config.result_dir, exist_ok=True)
    logger = setup_logger(
        name="Search2Adapt",
        log_dir=config.result_dir,
        level="INFO",
    )

    summary = print_config_summary(config)
    logger.info(summary)

    total_start = time.time()
    raw_organs = target_organs

    # --- Shared memory (created FIRST) ---
    memory = _make_shared_memory(config, logger)
    logger.info(f"Shared memory: {memory.unified_file}")

    # ---- Phase 1a: Perceptor ----
    logger.info("\n" + "#" * 60)
    logger.info("PHASE 1a: Perceptor — Domain Analysis")
    logger.info("#" * 60)

    perceptor = Perceptor(config, memory=memory, logger=logger)
    adaptation_plan = perceptor.run(raw_organs)

    # ---- Phase 1b: Strategist — plan generation ----
    logger.info("\n" + "#" * 60)
    logger.info("PHASE 1b: Strategist — Adaptation Plan Generation")
    logger.info("#" * 60)

    strategist = Strategist(config, memory=memory, logger=logger)
    strategist.set_domain_context(
        target_organs=list(adaptation_plan.get("text_prompts", {}).keys()),
        organ_label_map=adaptation_plan["organ_label_map"],
    )
    plans = strategist.generate_adaptation_plans(
        perceptor_plan=adaptation_plan,
        num_plans=num_plans,
    )
    logger.info(f"Generated {len(plans)} adaptation plans")

    # ---- Phase 2: MCTS Search ----
    logger.info("\n" + "#" * 60)
    logger.info("PHASE 2: MCTS Search — Pseudo-label Optimization")
    logger.info("#" * 60)

    from search2adapt.utils import scan_nnunet_dataset

    image_paths, case_ids = scan_nnunet_dataset(config.target_data_path)
    logger.info(f"Found {len(image_paths)} cases in {config.target_data_path}")

    if mcts_num_cases > 0 and len(image_paths) > mcts_num_cases:
        image_paths = image_paths[:mcts_num_cases]
        case_ids = case_ids[:mcts_num_cases]
        logger.info(f"Limiting to {mcts_num_cases} cases for MCTS search")

    if not image_paths:
        logger.error("No images found. Aborting.")
        sys.exit(1)

    # Reuse the memory created for phase 1 so that all three phases write to
    # the same memory_<domain>.json file.
    mcts_results = run_batch_search(
        config=config,
        adaptation_plan=adaptation_plan,
        image_paths=image_paths,
        case_ids=case_ids,
        initial_method=initial_method,
        memory=memory,
    )

    successful = sum(1 for v in mcts_results.values() if v is not None)
    rewards = [v[2] for v in mcts_results.values() if v is not None]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    logger.info("\nMCTS Search Summary:")
    logger.info(f"  Total cases:      {len(case_ids)}")
    logger.info(f"  Successful:       {successful}")
    logger.info(f"  Average reward:   {avg_reward:.3f}")

    # ---- Save final pseudo-labels ----
    logger.info("\n" + "#" * 60)
    logger.info("Saving final pseudo-labels")
    logger.info("#" * 60)
    save_final_pseudolabels(mcts_results, config.result_dir, logger)

    # ---- Phase 3: Distillation ----
    if skip_distillation:
        logger.info("\nPHASE 3: Skipped (skip_distillation=True)")
    else:
        logger.info("\n" + "#" * 60)
        logger.info("PHASE 3: Knowledge Distillation → nnUNet")
        logger.info("#" * 60)

        builder = DatasetBuilder(config, logger)
        dataset_path = builder.build_dataset(
            mcts_results=mcts_results,
            adaptation_plan=adaptation_plan,
            min_reward=min_reward,
        )

        runner = DistillationRunner(config, logger)
        dist_result = runner.run_distillation(dataset_path=dataset_path)
        logger.info(f"Distillation result: {json.dumps(dist_result, indent=2)}")

    total_elapsed = time.time() - total_start
    logger.info(f"\nPipeline complete in {total_elapsed / 60:.1f} minutes")
    logger.info(f"Results saved to: {config.result_dir}")

    return {"result_dir": config.result_dir, "adaptation_plan": adaptation_plan}


# ================================================================
#  Final pseudo-label saving
# ================================================================


def save_final_pseudolabels(
    mcts_results: Dict,
    result_dir: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Copy the best mask for each case into result_dir/final_pseudolabels/{case_id}.ext.
    """
    import os
    import shutil

    out_dir = os.path.join(result_dir, "final_pseudolabels")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for case_id, result in mcts_results.items():
        if result is None:
            if logger:
                logger.warning(f"  {case_id}: MCTS failed — no final mask")
            continue

        state, traj, reward = result
        src_path = state.mask_path

        if not src_path or not os.path.exists(src_path):
            if logger:
                logger.warning(f"  {case_id}: mask path invalid: {src_path}")
            continue

        dst_path = os.path.join(out_dir, f"{case_id}.nii.gz")

        try:
            shutil.copy2(src_path, dst_path)
            saved += 1
            if logger:
                logger.info(
                    f"  {case_id}: reward={reward:.3f}  "
                    f"mask → final_pseudolabels/{case_id}.nii.gz"
                )
        except Exception as e:
            if logger:
                logger.error(f"  {case_id}: copy failed: {e}")

    if logger:
        logger.info(
            f"Saved {saved}/{len(mcts_results)} final pseudo-labels to {out_dir}"
        )

    return out_dir


# Global variable to track the main process PID (set when signal handlers are registered)
_MAIN_PROCESS_PID: int = 0


def _cleanup_child_processes(signum, frame):
    """
    Signal handler for SIGINT/SIGTERM.
    Recursively terminates all child processes to prevent orphaned subprocesses.

    This addresses the issue where Ctrl+C doesn't kill subprocesses like:
    - SAM3 VFM inference (up to 30 min)
    - BiomedParse VFM inference (up to 30 min)
    - nnUNet training (hours)
    - ThreadPoolExecutor workers

    IMPORTANT: Only executes in the main process. Worker processes (e.g., from
    multiprocessing.Pool) inherit this handler but should skip cleanup.
    """
    import psutil

    # Only execute cleanup in the main process that registered the handlers
    # Worker processes (e.g., multiprocessing.Pool workers) should skip this
    global _MAIN_PROCESS_PID
    if _MAIN_PROCESS_PID and os.getpid() != _MAIN_PROCESS_PID:
        return  # Worker process - do nothing, let parent handle it

    # Safe logger access - get it fresh each time to avoid unbound variable issues
    _logger = logging.getLogger("Search2Adapt")

    # Log signal receipt
    sig_name = (
        "SIGINT"
        if signum == signal.SIGINT
        else "SIGTERM"
        if signum == signal.SIGTERM
        else str(signum)
    )
    _logger.warning(
        f"Received signal {sig_name} ({signum}), cleaning up child processes..."
    )

    try:
        current = psutil.Process()
        children = current.children(recursive=True)

        if children:
            _logger.info(f"Found {len(children)} child processes to terminate")

            # Strategy: For each child, first try to kill its entire process group
            # (catches GPU subprocesses that may have forked), then individual kill

            # Step 1: Try process group termination for processes that might have groups
            for child in children:
                try:
                    # Try to get process group ID and kill the entire group
                    pgid = os.getpgid(child.pid)
                    if (
                        pgid != child.pid
                    ):  # Has its own process group (leader forked children)
                        try:
                            os.killpg(pgid, signal.SIGTERM)
                            _logger.debug(f"Sent SIGTERM to process group {pgid}")
                        except (ProcessLookupError, PermissionError):
                            pass  # Process group already gone or can't signal
                except (ProcessLookupError, OSError):
                    pass  # Child process already gone

            # Step 2: Individual SIGTERM
            for child in children:
                try:
                    if child.is_running():
                        child.terminate()
                except psutil.NoSuchProcess:
                    pass  # Already gone

            # Wait up to 3 seconds for graceful shutdown
            gone, alive = psutil.wait_procs(children, timeout=3)
            still_alive = []  # Initialize for use in final message

            if alive:
                _logger.warning(
                    f"{len(alive)} processes still alive after SIGTERM, sending SIGKILL..."
                )

                # Step 3: Process group SIGKILL for survivors
                for p in alive:
                    try:
                        pgid = os.getpgid(p.pid)
                        if pgid != p.pid:
                            try:
                                os.killpg(pgid, signal.SIGKILL)
                            except (ProcessLookupError, PermissionError):
                                pass
                    except (ProcessLookupError, OSError):
                        pass

                # Step 4: Individual SIGKILL
                for p in alive:
                    try:
                        if p.is_running():
                            p.kill()
                    except psutil.NoSuchProcess:
                        pass

                # Wait another 2 seconds for force kill
                gone2, still_alive = psutil.wait_procs(alive, timeout=2)
                gone.extend(gone2)

                if still_alive:
                    _logger.error(
                        f"Failed to kill {len(still_alive)} processes: "
                        f"{[p.pid for p in still_alive]}"
                    )

            msg = (
                f"Terminated {len(gone)} processes, "
                f"force-killed {len(alive) - len(still_alive) if alive else 0} processes"
            )
            _logger.info(msg)
        else:
            _logger.info("No child processes to clean up")

    except Exception as e:
        _logger.error(f"Error during cleanup: {e}")

    # Exit with error code
    sys.exit(1)


if __name__ == "__main__":
    # ========================================================================
    #  Signal Handlers: Ensure clean shutdown on Ctrl+C / SIGTERM
    # ========================================================================
    # Register signal handlers to kill all child processes (subprocesses,
    # ThreadPoolExecutor threads, etc.) when the main process is interrupted.
    # This prevents orphaned GPU processes from continuing after Ctrl+C.
    # ========================================================================
    # Record main process PID before registering handlers
    # This prevents multiprocessing.Pool workers from triggering cleanup
    _MAIN_PROCESS_PID = os.getpid()
    signal.signal(signal.SIGINT, _cleanup_child_processes)
    signal.signal(signal.SIGTERM, _cleanup_child_processes)

    from search2adapt.cli import main as cli_main

    sys.exit(cli_main())

"""
Search2Adapt MCTS Search Loop Orchestrator
CoTS (Collaborative Tree Search) — Option B Hybrid Architecture

流程 (per case):
  Phase 0:   Evaluate initial state → early exit if ACCEPT
  Phase 0.5: Try memory priors (plan dialogues + trajectory priors) → shortcut
  Phase 1:   Collaborative Tree Search (plan-level, NO execution)
             - select → expand (Strategist proposes N plan variants)
             → evaluate (Evaluator critiques plan) → backprop
  Phase 2:   Execute top-K plans from tree with viability monitoring
             - Execute each plan step-by-step
             - Evaluate actual mask quality → record execution result
             - Return best executed plan

对应论文:
  - CoTS (Collaborative Tree Search): Planner-Critic 对话式搜索
  - Option B: Plan-level tree search + selective execution verification
"""

import concurrent.futures
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

from search2adapt.agents.base import Action, EvalResult, MaskState
from search2adapt.agents.evaluator import Evaluator
from search2adapt.agents.executor import Executor
from search2adapt.agents.strategist import Strategist
from search2adapt.config import Search2AdaptConfig
from search2adapt.mcts.tree import (
    Dialogue,
    PlanTree,
    RefinementPlan,
)
from search2adapt.memory import StrategyMemory


class MCTSSearchLoop:
    """
    MCTS 搜索主循环 — CoTS Option B Hybrid Architecture

    Per-case 搜索流程:
      Input:  initial MaskState (image + coarse pseudo-label)
      Output: best MaskState (refined pseudo-label) + best trajectory

    搜索策略:
      Phase 0:   Evaluate initial state → early exit if ACCEPT
      Phase 0.5: Try memory prior paths (shortcut if high reward)
      Phase 1:   Plan-level Collaborative Tree Search (no execution)
      Phase 2:   Execute top-K plans with viability monitoring
    """

    def __init__(
        self,
        config: Search2AdaptConfig,
        strategist: Strategist,
        executor: Executor,
        evaluator: Evaluator,
        memory: Optional[StrategyMemory] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.strategist = strategist
        self.executor = executor
        self.evaluator = evaluator
        self.memory = memory
        self.logger = logger or logging.getLogger("MCTSSearch")

    # =====================================================================
    #  Main entry: search for best pseudo-label
    # =====================================================================

    def search(self, initial_state: MaskState) -> Tuple[MaskState, List[Action], float]:
        """
        Run CoTS hybrid search for a single case.

        Args:
            initial_state: MaskState at step 0 (image + initial pseudo-label)

        Returns:
            (best_state, best_trajectory, best_reward)
        """
        case_id = initial_state.case_id
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Starting CoTS search for case: {case_id}")
        self.logger.info(f"{'=' * 60}")

        start_time = time.time()

        # ---- Phase 0: Evaluate initial state ----
        initial_eval = self.evaluator.run(
            initial_state, mode="text", action_taken="initial"
        )
        self.logger.info(
            f"Initial evaluation: reward={initial_eval.reward:.3f} "
            f"decision={initial_eval.decision}"
        )

        # Early exit if initial state is already good
        if initial_eval.decision == "ACCEPT":
            self.logger.info("Initial state accepted — no search needed")
            return initial_state, [], initial_eval.reward

        if not self.config.mcts.enable_refinement:
            self.logger.info(
                "Refinement disabled — keeping the raw VFM pseudo-label"
            )
            return initial_state, [], initial_eval.reward

        if not self.config.mcts.enable_mcts_search:
            return self._greedy_search(initial_state, initial_eval)

        # ---- Phase 0.5: Try memory prior paths ----
        prior_result = self._try_prior_paths(initial_state, initial_eval)
        if prior_result is not None:
            best_state, best_traj, best_reward = prior_result
            self.logger.info(
                f"Memory prior accepted! reward={best_reward:.3f} "
                f"path={[a.tool_id for a in best_traj]}"
            )
            return best_state, best_traj, best_reward
        # ---- Phase 1: Plan-level Collaborative Tree Search ----
        self.logger.info(f"\n{'~' * 40}")
        self.logger.info("Phase 1: Plan-level Collaborative Tree Search")
        self.logger.info(f"{'~' * 40}")

        plan_tree = PlanTree(
            root_state=initial_state,
            max_depth=self.config.mcts.plan_max_depth,
            exploration_weight=self.config.mcts.exploration_weight,
            diversity_weight=self.config.mcts.diversity_weight,
            logger=self.logger,
        )

        # Store initial eval context on root (do NOT backprop to avoid bias)
        plan_tree.root.plan_reward = initial_eval.reward

        plan_max_iter = self.config.mcts.plan_max_iterations
        plan_reward_thresh = self.config.mcts.plan_reward_threshold

        # Build eval history once per case — refreshed from memory before Phase 1
        eval_history = (
            self.memory.get_case_search_context(case_id)
            if self.memory is not None
            else ""
        )

        for iteration in range(plan_max_iter):
            self.logger.info(
                f"\n--- Plan Iteration {iteration + 1}/{plan_max_iter} ---"
            )

            # 1. Selection: UCT-guided descent to leaf
            leaf = plan_tree.select()
            self.logger.info(
                f"Selected plan node: depth={leaf.depth}, "
                f"visits={leaf.visits}, value={leaf.value:.3f}"
            )

            # Check depth limit
            if leaf.depth >= self.config.mcts.plan_max_depth:
                self.logger.info("Plan max depth reached, backpropagating leaf")
                plan_tree.backpropagate(leaf, leaf.plan_reward)
                continue

            # 2. Expansion: Strategist proposes N plan variants
            parent_plan_dict = leaf.plan.to_dict() if leaf.plan.steps else None
            evaluator_feedback = leaf.dialogue.evaluator_critique
            evaluator_suggestions = leaf.dialogue.evaluator_suggestions

            plan_variants = self.strategist.propose_plans(
                state=initial_state,
                parent_plan_dict=parent_plan_dict,
                evaluator_feedback=evaluator_feedback,
                evaluator_suggestions=evaluator_suggestions,
                num_variants=self.config.mcts.branching_factor,
                eval_history=eval_history,
            )

            if not plan_variants:
                self.logger.info("Strategist proposed no plan variants")
                plan_tree.backpropagate(leaf, leaf.plan_reward)
                continue

            # 3. Evaluate each plan variant (LLM-only, no execution) → create children
            children_with_rewards = []

            for variant in plan_variants:
                try:
                    # Convert steps to Action objects for RefinementPlan
                    plan_steps = [
                        Action(
                            tool_id=s["tool_id"],
                            params=s.get("params", {}),
                            description=variant.get("rationale", ""),
                        )
                        for s in variant["steps"]
                    ]

                    plan = RefinementPlan(
                        steps=plan_steps,
                        rationale=variant.get("rationale", ""),
                        target_defects=variant.get("target_defects", []),
                        expected_outcome=variant.get("expected_outcome", ""),
                    )

                    # Evaluator critiques the plan (no execution)
                    plan_reward, suggestions, critique = self.evaluator.plan_eval(
                        state=initial_state,
                        proposed_plan=plan.to_dict(),
                        eval_history=eval_history,
                        iteration=iteration,
                    )

                    # Create dialogue record
                    dialogue = Dialogue(
                        strategist_proposal=plan.rationale,
                        evaluator_critique=critique,
                        evaluator_reward=plan_reward,
                        evaluator_suggestions=suggestions,
                    )

                    # Create child node
                    child = leaf.add_child(
                        plan=plan,
                        dialogue=dialogue,
                        state=initial_state,  # Same state — no execution in Phase 1
                    )
                    child.plan_reward = plan_reward

                    self.logger.info(
                        f"  Plan [{','.join(s.tool_id for s in plan.steps)}]: "
                        f"plan_reward={plan_reward:.3f}"
                    )

                    children_with_rewards.append((child, plan_reward))

                except Exception as e:
                    self.logger.error(f"  Plan variant evaluation failed: {e}")
                    continue

            # 4. CoT Reflection: Strategist adjusts rewards if enabled
            if self.config.mcts.enable_cot_reflection and children_with_rewards:
                variant_list = [
                    plan_variants[i]
                    for i, (child, _) in enumerate(children_with_rewards)
                    if i < len(plan_variants)
                ]
                reward_list = [r for _, r in children_with_rewards]

                try:
                    reflections = self.strategist.reflect_on_evaluation(
                        state=initial_state,
                        plan_variants=variant_list,
                        eval_results=reward_list,
                    )
                    # Apply adjusted rewards to child nodes
                    for ref in reflections:
                        idx = ref["plan_id"]
                        if idx < len(children_with_rewards):
                            child, orig_reward = children_with_rewards[idx]
                            adj_reward = ref["adjusted_reward"]
                            child.plan_reward = adj_reward
                            children_with_rewards[idx] = (child, adj_reward)
                except Exception as e:
                    self.logger.warning(
                        f"CoT reflection failed (iteration {iteration}): {e} — "
                        "using original rewards"
                    )

            # 5. Backpropagation: propagate ALL children's rewards
            for child, child_reward in children_with_rewards:
                plan_tree.backpropagate(child, child_reward)

            # 5. Early stopping if plan reward threshold met
            if plan_tree.best_plan_reward >= plan_reward_thresh:
                self.logger.info(
                    f"Plan reward threshold met ({plan_tree.best_plan_reward:.3f} >= "
                    f"{plan_reward_thresh}), stopping plan search"
                )
                break

        # ---- Phase 2: Execute top-K plans ----
        self.logger.info(f"\n{'~' * 40}")
        self.logger.info("Phase 2: Execute top-K plans with viability monitoring")
        self.logger.info(f"{'~' * 40}")

        # Refresh eval_history — Phase 1 plan_evals have now been persisted to memory
        eval_history = (
            self.memory.get_case_search_context(case_id)
            if self.memory is not None
            else eval_history
        )

        top_k = self.config.mcts.top_k_execute
        exec_thresh = self.config.mcts.execution_threshold
        top_plan_nodes = plan_tree.get_top_k_plans(k=top_k)

        self.logger.info(f"Executing {len(top_plan_nodes)} top plans")

        for rank, plan_node in enumerate(top_plan_nodes):
            plan = plan_node.plan
            steps = plan.steps
            self.logger.info(
                f"\n  Executing plan {rank + 1}/{len(top_plan_nodes)}: "
                f"[{','.join(s.tool_id for s in steps)}] "
                f"(plan_reward={plan_node.plan_reward:.3f})"
            )

            state = initial_state
            executed_actions: List[Action] = []
            aborted = False

            for step_idx, action in enumerate(steps):
                try:
                    # Execute step
                    state = self.executor.run(state, action)
                    executed_actions.append(action)

                    self.logger.info(
                        f"    Step {step_idx + 1}/{len(steps)}: "
                        f"[{action.tool_id}] executed"
                    )

                    # Viability check (if enabled and not the last step)
                    if self.config.mcts.viability_check and step_idx < len(steps) - 1:
                        executed_dicts = [
                            {"tool_id": a.tool_id, "params": a.params}
                            for a in executed_actions
                        ]
                        remaining_dicts = [
                            {"tool_id": s.tool_id, "params": s.params}
                            for s in steps[step_idx + 1 :]
                        ]

                        viability_reward, still_viable = self.evaluator.viability_eval(
                            original_plan=plan.to_dict(),
                            executed_steps=executed_dicts,
                            current_state=state,
                            remaining_steps=remaining_dicts,
                            case_eval_history=eval_history,
                        )

                        if not still_viable:
                            self.logger.warning(
                                f"    Plan viability dropped to "
                                f"{viability_reward:.3f} — aborting plan"
                            )
                            aborted = True
                            break

                except Exception as e:
                    self.logger.error(f"    Step [{action.tool_id}] failed: {e}")
                    aborted = True
                    break

            # Evaluate final state (even if aborted, evaluate what we have)
            if executed_actions:
                action_chain = " → ".join(a.tool_id for a in executed_actions)
                eval_result = self.evaluator.run(
                    state, mode="text", action_taken=action_chain
                )
                execution_reward = eval_result.reward

                self.logger.info(
                    f"  Plan {rank + 1} execution result: "
                    f"reward={execution_reward:.3f} "
                    f"decision={eval_result.decision} "
                    f"aborted={aborted}"
                )

                # Record execution result on the plan node
                plan_tree.record_execution(
                    node=plan_node,
                    reward=execution_reward,
                    state=state,
                    eval_result=eval_result,
                )

                # Early exit if execution threshold met
                if execution_reward >= exec_thresh:
                    self.logger.info(
                        f"Execution threshold met ({execution_reward:.3f} >= "
                        f"{exec_thresh}), stopping execution"
                    )
                    break

        # ---- Extract results ----
        elapsed = time.time() - start_time
        plan_stats = plan_tree.get_stats()

        self.logger.info(f"\nCoTS Search Complete for {case_id}")
        self.logger.info(f"  Plan iterations: {plan_stats['total_iterations']}")
        self.logger.info(f"  Total plan nodes: {plan_stats['total_nodes']}")
        self.logger.info(f"  Max depth: {plan_stats['max_depth_reached']}")
        self.logger.info(f"  Best plan reward: {plan_stats['best_plan_reward']:.3f}")
        self.logger.info(
            f"  Best execution reward: {plan_stats['best_execution_reward']:.3f}"
        )
        self.logger.info(f"  Executed plans: {plan_stats['executed_plans']}")
        self.logger.info(f"  Time: {elapsed:.1f}s")

        # Get best results (Phase 2 execution preferred over Phase 1 plan)
        best_trajectory = plan_tree.get_best_plan_trajectory()
        best_state = plan_tree.get_best_state() or initial_state
        best_reward = plan_tree.get_best_reward()

        # If no plan was executed or all failed, fall back to initial state
        if best_reward <= 0.0:
            best_state = initial_state
            best_trajectory = []
            best_reward = initial_eval.reward

        # Write successful trajectory to memory
        if self.memory and best_reward >= self.memory.prior_min_reward:
            self.memory.add_trajectory(
                actions=best_trajectory,
                reward=best_reward,
                case_id=case_id,
            )

        # Write successful plan dialogue to memory (CoTS extended memory)
        best_exec_node = plan_tree.best_execution_node
        if (
            self.memory
            and best_exec_node is not None
            and best_exec_node.execution_reward is not None
            and best_exec_node.execution_reward >= self.memory.prior_min_reward
        ):
            self.memory.add_plan_dialogue(
                plan_steps=[
                    {"tool_id": s.tool_id, "params": s.params}
                    for s in best_exec_node.plan.steps
                ],
                strategist_proposal=best_exec_node.dialogue.strategist_proposal,
                evaluator_critique=best_exec_node.dialogue.evaluator_critique,
                plan_reward=best_exec_node.plan_reward,
                execution_reward=best_exec_node.execution_reward,
                case_id=case_id,
            )

        # Save search stats
        self._save_search_stats(case_id, plan_stats, best_trajectory, elapsed)

        return best_state, best_trajectory, best_reward

    def _greedy_search(
        self,
        initial_state: MaskState,
        initial_eval: EvalResult,
    ) -> Tuple[MaskState, List[Action], float]:
        best_plan = self.strategist.propose_plans(
            state=initial_state,
            parent_plan_dict=None,
            evaluator_feedback="",
            evaluator_suggestions=[],
            num_variants=1,
            eval_history="",
        )

        if not best_plan:
            self.logger.info("Greedy search: no plan proposed, returning initial state")
            return initial_state, [], initial_eval.reward

        steps = best_plan[0].get("steps", [])
        if not steps:
            return initial_state, [], initial_eval.reward

        actions = [
            Action(
                tool_id=s["tool_id"],
                params=s.get("params", {}),
                description=best_plan[0].get("rationale", ""),
            )
            for s in steps
        ]

        state = initial_state
        executed_actions: List[Action] = []

        for action in actions:
            try:
                state = self.executor.run(state, action)
                executed_actions.append(action)
                self.logger.info(f"Greedy step: [{action.tool_id}] executed")
            except Exception as e:
                self.logger.error(f"Greedy step [{action.tool_id}] failed: {e}")
                break

        if executed_actions:
            action_chain = " → ".join(a.tool_id for a in executed_actions)
            eval_result = self.evaluator.run(
                state, mode="text", action_taken=action_chain
            )
            self.logger.info(
                f"Greedy search complete: reward={eval_result.reward:.3f} "
                f"path={[a.tool_id for a in executed_actions]}"
            )
            return state, executed_actions, eval_result.reward

        return initial_state, [], initial_eval.reward

    # =====================================================================
    #  Phase 0.5: Try memory prior paths
    # =====================================================================

    def _try_prior_paths(
        self,
        initial_state: MaskState,
        initial_eval: EvalResult,
    ) -> Optional[Tuple[MaskState, List[Action], float]]:
        """
        Try executing top memory prior paths directly.
        If any achieves reward >= prior_accept_threshold, return it.
        """
        if self.memory is None:
            return None

        prior_paths = self.strategist.get_prior_actions()
        if not prior_paths:
            self.logger.info("No memory priors available")
            return None

        accept_thresh = self.memory.prior_accept_threshold
        self.logger.info(
            f"Trying {len(prior_paths)} memory prior paths "
            f"(accept threshold={accept_thresh})"
        )

        for i, action_sequence in enumerate(prior_paths):
            self.logger.info(
                f"  Prior path {i + 1}: {[a.tool_id for a in action_sequence]}"
            )

            state = initial_state
            success = True
            executed_actions = []

            for action in action_sequence:
                try:
                    state = self.executor.run(state, action)
                    executed_actions.append(action)
                except Exception as e:
                    self.logger.warning(
                        f"  Prior path {i + 1} failed at [{action.tool_id}]: {e}"
                    )
                    success = False
                    break

            if not success:
                continue

            # Evaluate final state
            prior_action_chain = " → ".join(a.tool_id for a in executed_actions)
            eval_result = self.evaluator.run(
                state, mode="text", action_taken=prior_action_chain
            )
            self.logger.info(f"  Prior path {i + 1} reward: {eval_result.reward:.3f}")

            if eval_result.reward >= accept_thresh:
                # Update memory count
                self.memory.add_trajectory(
                    actions=executed_actions,
                    reward=eval_result.reward,
                    case_id=initial_state.case_id,
                )
                return state, executed_actions, eval_result.reward

        self.logger.info("No prior path met acceptance threshold")
        return None

    # =====================================================================
    #  Utilities
    # =====================================================================

    def _save_search_stats(
        self,
        case_id: str,
        stats: Dict,
        best_trajectory: List[Action],
        elapsed: float,
    ):
        """Save search statistics to JSON"""
        stats_dir = os.path.join(self.config.result_dir, case_id)
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(stats_dir, "mcts_stats.json")

        output = {
            "case_id": case_id,
            "elapsed_seconds": round(elapsed, 1),
            "tree_stats": stats,
            "best_trajectory": [a.to_dict() for a in best_trajectory],
        }

        with open(stats_path, "w") as f:
            json.dump(output, f, indent=2)


# =====================================================================
#  Batch search: run MCTS for all cases in dataset
# =====================================================================


def run_batch_search(
    config: Search2AdaptConfig,
    adaptation_plan: Dict,
    image_paths: List[str],
    case_ids: List[str],
    initial_method: str = "biomedparse",
    memory: Optional[StrategyMemory] = None,
) -> Dict[str, Tuple[MaskState, List[Action], float]]:
    """Run MCTS search for all cases in a dataset."""
    # Named under the configured "Search2Adapt" logger so per-case records
    # reach its console and file handlers instead of vanishing into an
    # unconfigured branch of the logging tree.
    logger = logging.getLogger("Search2Adapt.BatchSearch")

    # Reuse injected memory or create a new one
    if memory is None:
        memory = StrategyMemory(
            memory_dir=os.path.join(config.result_dir, "memory"),
            domain_key=config.experiment_name,
            max_trajectories=config.memory.max_trajectories,
            prior_top_k=config.memory.prior_top_k,
            prior_min_reward=config.memory.prior_min_reward,
            prior_accept_threshold=config.memory.prior_accept_threshold,
            logger=logger,
        )

    # Process each case (parallel if max_workers > 1)
    results: Dict = {}
    results_lock = threading.Lock()
    max_workers = getattr(config, "max_workers", 1)

    def _process_case(img_path: str, case_id: str, idx: int):
        """Process a single case in its own thread with dedicated agent instances."""
        case_logger = logging.getLogger(f"Search2Adapt.BatchSearch.{case_id}")
        case_logger.info(f"\n{'#' * 60}")
        case_logger.info(f"Processing case {idx + 1}/{len(case_ids)}: {case_id}")
        case_logger.info(f"{'#' * 60}")

        try:
            case_evaluator = Evaluator(
                config,
                memory=memory,
                logger=case_logger,
            )
            case_evaluator.set_domain_context(
                target_organs=list(adaptation_plan.get("text_prompts", {}).keys()),
                organ_label_map=adaptation_plan["organ_label_map"],
                modality=adaptation_plan["domain_context"]["modality"],
                anatomical_site=adaptation_plan["domain_context"]["anatomical_site"],
            )
            case_strategist = Strategist(config, memory=memory, logger=case_logger)
            case_strategist.set_domain_context(
                target_organs=list(adaptation_plan.get("text_prompts", {}).keys()),
                organ_label_map=adaptation_plan["organ_label_map"],
            )
            case_executor = Executor(
                config, adaptation_plan=adaptation_plan, logger=case_logger
            )
            case_search_loop = MCTSSearchLoop(
                config=config,
                strategist=case_strategist,
                executor=case_executor,
                evaluator=case_evaluator,
                memory=memory,
                logger=case_logger,
            )

            # Generate initial pseudo-label
            initial_state = case_executor.generate_initial_pseudolabel(
                image_path=img_path,
                case_id=case_id,
                method=initial_method,
            )

            # Run MCTS search
            best_state, best_traj, best_reward = case_search_loop.search(initial_state)
            with results_lock:
                results[case_id] = (best_state, best_traj, best_reward)

            case_logger.info(
                f"Case {case_id}: reward={best_reward:.3f}, "
                f"path={[a.tool_id for a in best_traj]}"
            )

        except Exception as e:
            case_logger.error(f"Case {case_id} failed: {e}", exc_info=True)
            with results_lock:
                if 'initial_state' in locals() and initial_state is not None:
                    case_logger.info(f"Falling back to initial pseudo-label for case {case_id}.")
                    results[case_id] = (initial_state, [], 0.0)
                else:
                    results[case_id] = None

        finally:
            # Flush any pending memory writes at the end of each case
            memory.flush()

    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_process_case, img_path, case_id, i): case_id
                for i, (img_path, case_id) in enumerate(zip(image_paths, case_ids))
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()  # re-raise any unhandled exceptions
    else:
        for i, (img_path, case_id) in enumerate(zip(image_paths, case_ids)):
            _process_case(img_path, case_id, i)

    # Save summary
    summary = {
        "total_cases": len(case_ids),
        "successful": sum(1 for v in results.values() if v is not None),
        "results": {},
    }
    for case_id, result in results.items():
        if result is not None:
            state, traj, reward = result
            summary["results"][case_id] = {
                "reward": reward,
                "trajectory": [a.to_dict() for a in traj],
                "mask_path": state.mask_path,
            }
        else:
            summary["results"][case_id] = {"error": "failed"}

    summary_path = os.path.join(config.result_dir, "search_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nBatch search complete. Summary saved to {summary_path}")
    logger.info(f"Memory stats: {memory.get_stats()}")

    return results

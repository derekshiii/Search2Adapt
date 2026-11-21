"""
Search2Adapt Strategist Agent (Planner)

职责 (两个互补的核心功能):

  A. Adaptation Plan Generation  (Phase 1 — 在 Perceptor 完成后调用一次)
     ------------------------------------------------------------------
     基于 Perceptor 感知结果 (domain_context + image_profile), 调用 LLM
     生成 N 条完整的、互补的 Adaptation Plans (工具链候选集).
     Plans 按 priority 排序后写入 StrategyMemory → adaptation_plans_<domain>.json.

     API: strategist.generate_adaptation_plans(perceptor_plan)
          → List[AdaptationPlan]  (也已写入 memory)

  B. MCTS Expansion  (Phase 2 — 每次 MCTS 迭代的 Expansion 步骤)
     ------------------------------------------------------------------
     接收当前节点状态 + Evaluator 反馈 + Memory 先验, 调用 LLM 推理,
     提出 2-3 个候选 *下一步* 动作 (candidate actions).
     每个候选动作成为 MCTS 树的一个新子节点.

     API: strategist.run(state, eval_result, action_history)
          → List[Action]  (候选动作列表)

  辅助功能:
     strategist.select_plan(perceptor_plan)
       → 如果 memory 中已有 plans, 直接返回已有 plans (跳过 LLM 调用).
         否则调用 generate_adaptation_plans().

     strategist.get_prior_actions()
       → 将 memory 中的高优先级 plan steps 转为 Action 列表 (供 MCTS Phase 0.5).

对应论文:
  - Strategist (Planner): 系统的核心大脑
  - 维护 MCTS 搜索树
  - 根据 Perceptor 感知内容生成并保存 adaptation plan 到 memory
  - 面对新样本时基于 memory prior 快速选择 plan
"""

import json
import logging
import re
from typing import Dict, List, Optional

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


from search2adapt.agents.base import (
    Action,
    BaseAgent,
    EvalResult,
    LLMCaller,
    MaskState,
    extract_mask_features,
    features_to_text,
)
from search2adapt.config import Search2AdaptConfig
from search2adapt.memory import AdaptationPlan, StrategyMemory
from search2adapt.prompts import PROMPT_REGISTRY
from search2adapt.toolbox.tools import (
    TOOL_REGISTRY,
    get_filtered_tool_registry,
    get_tool_descriptions_for_llm_filtered,
)


class Strategist(BaseAgent):
    """
    Strategist (Planner) Agent

    状态:
      target_organs:      Perceptor 校正后的目标器官列表
      organ_label_map:    {organ_name: label_id}
      _toolbox_desc:      工具描述 (静态, 从 TOOL_REGISTRY 生成)
      memory:             StrategyMemory (可选, 用于读写 plans 和 trajectories)
    """

    def __init__(
        self,
        config: Search2AdaptConfig,
        memory: Optional[StrategyMemory] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(name="Strategist", config=config, logger=logger)
        self.llm_caller = LLMCaller(config.llm, self.logger)
        self.memory = memory

        allowed_categories = config.toolbox.get_allowed_categories()
        self._toolbox_desc = get_tool_descriptions_for_llm_filtered(allowed_categories)
        self._filtered_tool_registry = get_filtered_tool_registry(allowed_categories)

        self.target_organs: List[str] = []
        self.organ_label_map: Dict[str, int] = {}

    def set_domain_context(
        self,
        target_organs: List[str],
        organ_label_map: Dict[str, int],
    ):
        """Called after Perceptor finishes to inject domain knowledge."""
        self.target_organs = target_organs
        self.organ_label_map = organ_label_map

    # =====================================================================
    #  A. Adaptation Plan Generation
    # =====================================================================

    def generate_adaptation_plans(
        self,
        perceptor_plan: Dict,
        num_plans: int = 4,
        force_regenerate: bool = False,
        case_image_profile: Optional[Dict] = None,
    ) -> List[AdaptationPlan]:
        """
        Generate N complete Adaptation Plans from the Perceptor domain analysis.

        This method should be called ONCE after Perceptor finishes, before MCTS.
        The generated plans are persisted to memory so that subsequent cases in
        the same domain can skip LLM generation entirely (select_plan() handles this).

        Args:
            perceptor_plan:      Full output dict from Perceptor.run()
            num_plans:           Number of diverse plans to generate (default 4)
            force_regenerate:    If True, regenerate even if memory already has plans
            case_image_profile:  Optional per-case profile dict from
                                 Perceptor.get_case_profile(). When provided, the
                                 prompt includes case-specific stats so the LLM can
                                 calibrate tool parameters (sigma, min_size, etc.)
                                 to the individual case rather than domain averages.

        Returns:
            List[AdaptationPlan] sorted by priority
        """
        # 1. If memory already has plans and we're not forcing, return cached plans.
        #    Exception: when a per-case profile is supplied we always regenerate so
        #    that tool parameters are calibrated to the individual case rather than
        #    domain averages — this is the primary fix for plan rigidity.
        if (
            not force_regenerate
            and case_image_profile is None
            and self.memory is not None
            and self.memory.adaptation_plans
        ):
            self.logger.info(
                f"[Strategist] Using {len(self.memory.adaptation_plans)} cached "
                f"adaptation plans from memory"
            )
            return self.memory.get_plans_by_priority()

        # 2. Build prompt components from perceptor output
        if self.config.mcts.enable_perceptor_context:
            domain_context_text = self._format_domain_context(perceptor_plan)
            image_profile_text = self._format_image_profile(perceptor_plan)
            case_profile_text = self._format_case_image_profile(case_image_profile)
        else:
            # w/o Perceptor ablation: plan without any domain-specific
            # characteristics, so the Strategist falls back to blind planning.
            # The organ list comes from the user request, not from perception,
            # so it stays — otherwise the ablation changes the task itself.
            organs = ", ".join(self.target_organs)
            domain_context_text = (
                f"Target organs: {organs}\n"
                "(Domain profile DISABLED for ablation)"
            )
            image_profile_text = "(Domain profile DISABLED for ablation)"
            case_profile_text = "(Domain profile DISABLED for ablation)"
        memory_priors_text = self._build_memory_priors()

        prompt = PROMPT_REGISTRY["STRATEGIST_PLAN_GENERATE"].format(
            domain_context=domain_context_text,
            image_profile=image_profile_text,
            case_image_profile=case_profile_text,
            toolbox_description=self._toolbox_desc,
            memory_priors=memory_priors_text,
            num_plans=num_plans,
        )

        system_prompt = (
            "You are an expert medical image processing strategist specializing in "
            "domain adaptation. Design diverse, complementary adaptation plans. "
            "Respond ONLY with valid JSON — no markdown fences, no prose outside JSON."
        )

        # 3. Call LLM
        plans: List[AdaptationPlan] = []
        try:
            response = self.llm_caller.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.7,
            )
            plans = self._parse_plans_response(response, num_plans)
            self.logger.info(
                f"[Strategist] Generated {len(plans)} adaptation plans via LLM"
            )
        except Exception as e:
            self.logger.error(
                f"[Strategist] LLM plan generation failed: {e}. "
                f"Using rule-based fallback plans."
            )
            plans = self._fallback_plans(perceptor_plan, num_plans)

        # 4. Persist to memory (also promotes top trajectories)
        if self.memory is not None:
            self.memory.set_adaptation_plans(plans)
            self.logger.info(
                f"[Strategist] Plans saved to memory "
                f"({len(self.memory.adaptation_plans)} total after promotion)"
            )

        # Return final list (may include promoted trajectory plans)
        if self.memory is not None:
            return self.memory.get_plans_by_priority()
        return sorted(plans, key=lambda p: p.priority)

    def select_plan(
        self,
        perceptor_plan: Dict,
        num_plans: int = 4,
        case_image_profile: Optional[Dict] = None,
    ) -> List[AdaptationPlan]:
        """
        Return adaptation plans for this domain.

        When ``case_image_profile`` is provided (per-case stats from
        Perceptor.get_case_profile()), cached domain-level plans are bypassed
        and fresh per-case plans are generated so tool parameters are calibrated
        to the individual case.

        If no per-case profile is provided and memory already has plans, returns
        them directly (no LLM call).

        Args:
            perceptor_plan:      Full output dict from Perceptor.run()
            num_plans:           Number of plans to generate if cache is empty
            case_image_profile:  Optional per-case profile dict. When given,
                                 always generates fresh plans (bypasses cache).

        Returns:
            List[AdaptationPlan] sorted by priority (highest first)
        """
        # Per-case profile supplied → always generate fresh, case-specific plans
        if case_image_profile is not None:
            self.logger.info(
                "[Strategist] select_plan: per-case profile provided — "
                "generating fresh case-specific plans (bypassing cache)"
            )
            return self.generate_adaptation_plans(
                perceptor_plan,
                num_plans=num_plans,
                force_regenerate=True,
                case_image_profile=case_image_profile,
            )

        # No per-case profile → use cached domain plans if available
        if self.memory is not None and self.memory.adaptation_plans:
            cached = self.memory.get_plans_by_priority()
            self.logger.info(
                f"[Strategist] select_plan: returning {len(cached)} plans from memory "
                f"(no LLM call needed)"
            )
            return cached

        self.logger.info(
            "[Strategist] select_plan: no cached plans found, generating..."
        )
        return self.generate_adaptation_plans(perceptor_plan, num_plans=num_plans)

    # =====================================================================
    #  A2. CoTS Phase 1 — Propose complete plan variants for tree search
    # =====================================================================

    def propose_plans(
        self,
        state: MaskState,
        parent_plan_dict: Optional[Dict] = None,
        evaluator_feedback: str = "",
        evaluator_suggestions: Optional[List[str]] = None,
        num_variants: int = 3,
        eval_history: str = "",
    ) -> List[Dict]:
        """
        Propose N complete refinement plan variants for CoTS Phase 1.

        Called during PlanTree expansion.  The Strategist generates diverse
        plan variants that build on the parent plan and incorporate the
        Evaluator's critique / suggestions.

        Args:
            state:                Current MaskState (for context).
            parent_plan_dict:     Parent plan as dict (``RefinementPlan.to_dict()``),
                                  or None for root expansion.
            evaluator_feedback:   Evaluator's critique text on the parent plan.
            evaluator_suggestions: Evaluator's improvement suggestions (list of str).
            num_variants:         Number of diverse plan variants to generate.
            eval_history:         Formatted eval history string from memory
                                  (from memory.get_case_search_context). Injected
                                  into the prompt so the Planner avoids proposing
                                  approaches already shown to fail.

        Returns:
            List of plan variant dicts, each with keys:
              ``steps``, ``rationale``, ``target_defects``, ``expected_outcome``
        """
        # 1. Build state description
        state_desc = self._build_state_description(state, action_history=None)

        # 2. Domain context — enriched with domain_profile_summary from memory.
        #    The organ list / label map come from the user's request, not from
        #    the Perceptor, so they stay in every ablation; only the perceived
        #    domain profile is suppressed for w/o Perceptor.
        if not self.config.mcts.enable_perceptor_context:
            domain_profile_summary = "(Domain profile DISABLED for ablation)"
        elif self.memory is not None:
            domain_profile_summary = self.memory.get_domain_context_summary()
        else:
            domain_profile_summary = "(Domain profile not available)"
        domain_context = (
            f"Target organs: {', '.join(self.target_organs)}\n"
            f"Organ label map: {json.dumps(self.organ_label_map)}"
        )

        # 3. Eval history fallback
        if not eval_history:
            eval_history = (
                self.memory.get_case_search_context(state.case_id)
                if self.memory is not None
                else "(No evaluation history available)"
            )

        # 4. Parent plan text
        if parent_plan_dict and parent_plan_dict.get("steps"):
            parent_plan_text = json.dumps(
                parent_plan_dict, indent=2, ensure_ascii=False
            )
        else:
            parent_plan_text = "(No parent plan — this is the root expansion)"

        # 5. Evaluator feedback text
        if evaluator_feedback:
            eval_text = evaluator_feedback
            if evaluator_suggestions:
                eval_text += "\n\nSuggestions for improvement:\n"
                for i, s in enumerate(evaluator_suggestions, 1):
                    eval_text += f"  {i}. {s}\n"
        else:
            eval_text = "(No evaluator feedback yet — initial plan proposal)"

        # 6. Plan dialogue priors from memory
        plan_dialogue_priors = self._build_plan_dialogue_priors()

        # 7. Fill prompt template
        prompt = PROMPT_REGISTRY["STRATEGIST_PLAN_PROPOSAL"].format(
            domain_profile_summary=domain_profile_summary,
            eval_history=eval_history,
            state_description=state_desc,
            domain_context=domain_context,
            toolbox_description=self._toolbox_desc,
            parent_plan=parent_plan_text,
            evaluator_feedback=eval_text,
            plan_dialogue_priors=plan_dialogue_priors,
            num_variants=num_variants,
        )

        system_prompt = (
            "You are the Planner in a Collaborative Tree Search framework for "
            "medical image segmentation. Propose diverse refinement plan "
            "variants. Respond ONLY with valid JSON — no markdown fences, "
            "no prose outside JSON."
        )

        try:
            response = self.llm_caller.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.7,  # Higher diversity for plan variants
            )
            plans = self._parse_plan_variants_response(response)
        except Exception as e:
            self.logger.error(
                f"[Strategist] propose_plans LLM call failed: {e}. "
                f"Using fallback plan variants."
            )
            plans = self._fallback_plan_variants(state, num_variants)

        # Post-process: remove structural duplicates, then mutate homogeneous params
        plans = self._deduplicate_variants(plans, parent_plan_dict)
        plans = self._mutate_params(plans)

        self.logger.info(
            f"[Strategist] propose_plans: case={state.case_id} "
            f"generated {len(plans)} plan variants"
        )
        return plans

    def _parse_plan_variants_response(self, response: str) -> List[Dict]:
        """Parse LLM JSON response for plan variants (STRATEGIST_PLAN_PROPOSAL)."""
        clean = self._clean_json_text(response)
        clean = self._sanitize_json_text(clean)
        clean = self._extract_json_object(clean)

        data = _loads_robust(clean)

        # Log CoT reasoning from analysis field
        analysis = data.get("analysis", "")
        if analysis:
            self.logger.debug(
                f"[Strategist] propose_plans CoT analysis: {analysis[:300]}"
            )

        raw_variants = data.get("plan_variants", [])

        plans = []
        for raw in raw_variants:
            steps = raw.get("steps", [])
            valid_steps = []
            for step in steps:
                tool_id = step.get("tool_id", "").strip().lower()
                if tool_id in TOOL_REGISTRY:
                    valid_steps.append(
                        {"tool_id": tool_id, "params": step.get("params", {})}
                    )
                else:
                    self.logger.warning(
                        f"[Strategist] Unknown tool_id '{tool_id}' in plan variant, "
                        f"skipping step"
                    )

            if not valid_steps:
                self.logger.warning(
                    "[Strategist] Plan variant has no valid steps, skipping"
                )
                continue

            plans.append(
                {
                    "steps": valid_steps,
                    "rationale": raw.get("rationale", ""),
                    "target_defects": raw.get("target_defects", []),
                    "expected_outcome": raw.get("expected_outcome", ""),
                }
            )

        return plans

    def _fallback_plan_variants(
        self, state: MaskState, num_variants: int
    ) -> List[Dict]:
        """Fallback plan variants when LLM is unavailable for propose_plans()."""
        variants = [
            {
                "steps": [
                    {
                        "tool_id": "postprocess_segmentation",
                        "params": {
                            "min_size": 100,
                            "keep_largest_only": True,
                            "fill_holes": True,
                        },
                    },
                ],
                "rationale": "Conservative cleanup: remove fragments and fill holes",
                "target_defects": ["fragments", "holes"],
                "expected_outcome": "Cleaner mask with fewer artifacts",
            },
            {
                "steps": [
                    {"tool_id": "morphological_closing", "params": {"kernel_size": 3}},
                    {
                        "tool_id": "postprocess_segmentation",
                        "params": {"keep_largest_only": True, "fill_holes": True},
                    },
                ],
                "rationale": "Close gaps then clean up",
                "target_defects": ["gaps", "holes", "fragments"],
                "expected_outcome": "Filled gaps with cleanup",
            },
            {
                "steps": [
                    {"tool_id": "gaussian_smooth_mask", "params": {"sigma": 1.0}},
                    {
                        "tool_id": "postprocess_segmentation",
                        "params": {"keep_largest_only": True, "fill_holes": True},
                    },
                ],
                "rationale": "Smooth boundaries then clean up",
                "target_defects": ["jagged_boundaries"],
                "expected_outcome": "Smoother boundaries",
            },
        ]
        return variants[:num_variants]

    def _build_plan_dialogue_priors(self) -> str:
        """Build plan dialogue prior text for CoTS plan proposal prompt."""
        if self.memory is None:
            return "No plan dialogue memory available."

        summaries = self.memory.get_plan_dialogue_summaries()
        if not summaries:
            return "No prior plan dialogues available for this domain."

        lines = ["Successful plan dialogues from similar cases:"]
        for i, s in enumerate(summaries):
            lines.append(f"  Plan {i + 1}: {s}")
        return "\n".join(lines)

    # =====================================================================
    #  A3. CoTS Phase 1 — CoT Reflection on Evaluator scores
    # =====================================================================

    def reflect_on_evaluation(
        self,
        state: MaskState,
        plan_variants: List[Dict],
        eval_results: List[float],
    ) -> List[Dict]:
        """
        Reflect on the Evaluator's critique rewards and optionally adjust them.

        Called in Phase 1 after the Evaluator scores each plan variant but BEFORE
        backpropagation.  The Strategist reasons about whether each score is
        calibrated correctly given domain knowledge and proposes small adjustments
        (±0.20 max).

        Args:
            state:         Current MaskState (for context).
            plan_variants: List of plan variant dicts (from propose_plans).
            eval_results:  List of float rewards from Evaluator.plan_eval(),
                           one per plan variant.

        Returns:
            List of reflection dicts, each with keys:
              ``plan_id``, ``original_reward``, ``adjusted_reward``, ``reasoning``
        """
        if not plan_variants or not eval_results:
            return []

        # 1. Build state description
        state_desc = self._build_state_description(state, action_history=None)

        # 2. Domain context
        domain_context = (
            f"Target organs: {', '.join(self.target_organs)}\n"
            f"Organ label map: {json.dumps(self.organ_label_map)}"
        )
        if self.config.mcts.enable_perceptor_context and self.memory is not None:
            domain_summary = self.memory.get_domain_context_summary()
            if domain_summary:
                domain_context = domain_summary + "\n" + domain_context

        # 3. Build plan-eval summary text
        lines = []
        for i, (plan, reward) in enumerate(zip(plan_variants, eval_results)):
            steps_desc = " → ".join(
                s.get("tool_id", "?") for s in plan.get("steps", [])
            )
            rationale = plan.get("rationale", "")
            lines.append(
                f"Plan {i} | reward={reward:.3f} | steps=[{steps_desc}] | "
                f"rationale={rationale}"
            )
        plan_eval_summary = "\n".join(lines)

        # 4. Fill prompt
        prompt = PROMPT_REGISTRY["STRATEGIST_REFLECTION"].format(
            domain_context=domain_context,
            state_description=state_desc,
            plan_eval_summary=plan_eval_summary,
        )

        system_prompt = (
            "You are the Planner in a Collaborative Tree Search framework. "
            "Reflect on the Evaluator's plan scores and adjust if needed. "
            "Respond ONLY with valid JSON — no markdown fences, no prose outside JSON."
        )

        try:
            response = self.llm_caller.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.3,
            )
            reflections = self._parse_reflection_response(
                response, plan_variants, eval_results
            )
        except Exception as e:
            self.logger.error(
                f"[Strategist] reflect_on_evaluation LLM call failed: {e}. "
                f"Returning original rewards unchanged."
            )
            reflections = [
                {
                    "plan_id": i,
                    "original_reward": r,
                    "adjusted_reward": r,
                    "reasoning": f"Reflection skipped (LLM error: {e})",
                }
                for i, r in enumerate(eval_results)
            ]

        self.logger.info(
            f"[Strategist] reflect_on_evaluation: case={state.case_id} "
            f"reflected on {len(reflections)} plan(s)"
        )
        for ref in reflections:
            delta = ref["adjusted_reward"] - ref["original_reward"]
            if abs(delta) > 0.01:
                self.logger.info(
                    f"[Strategist] Reflection plan_{ref['plan_id']}: "
                    f"{ref['original_reward']:.3f} → {ref['adjusted_reward']:.3f} "
                    f"(Δ={delta:+.3f})"
                )
        return reflections

    def _parse_reflection_response(
        self,
        response: str,
        plan_variants: List[Dict],
        eval_results: List[float],
    ) -> List[Dict]:
        """Parse LLM JSON response for CoT reflection (STRATEGIST_REFLECTION)."""
        clean = self._clean_json_text(response)
        clean = self._sanitize_json_text(clean)
        clean = self._extract_json_object(clean)

        data = _loads_robust(clean)
        raw_reflections = data.get("reflections", [])

        results = []
        for i, (plan, orig_reward) in enumerate(zip(plan_variants, eval_results)):
            # Find matching reflection by plan_id (index)
            match = next((r for r in raw_reflections if r.get("plan_id") == i), None)
            if match is None and i < len(raw_reflections):
                match = raw_reflections[i]  # fall back to positional match

            if match is not None:
                adj = float(match.get("adjusted_reward", orig_reward))
                # Enforce ±0.20 cap and [0, 1] clamp
                delta = adj - orig_reward
                delta = max(-0.20, min(0.20, delta))
                adj = max(0.0, min(1.0, orig_reward + delta))
                reasoning = match.get("reasoning", "")
            else:
                adj = orig_reward
                reasoning = "No reflection provided — keeping original reward"

            results.append(
                {
                    "plan_id": i,
                    "original_reward": orig_reward,
                    "adjusted_reward": adj,
                    "reasoning": reasoning,
                }
            )

        return results

    # =====================================================================
    #  Deduplication + parametric mutation helpers (Steps 8+9)
    # =====================================================================

    @staticmethod
    def _plan_signature(plan_dict: Dict) -> tuple:
        """Return a hashable signature of a plan for structural deduplication.

        Two plans are considered structurally identical if they share the same
        ordered sequence of tool_ids.  Parameters are NOT compared here so that
        parameter-only variants are permitted.
        """
        steps = plan_dict.get("steps", [])
        return tuple(s.get("tool_id", "") for s in steps)

    def _deduplicate_variants(
        self,
        variants: List[Dict],
        parent_plan_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """Remove plan variants that are structurally identical to the parent
        plan or to each other.

        A 'structurally identical' plan has the same ordered sequence of
        tool_ids.  Parameter-only differences are allowed.

        Args:
            variants:         List of plan variant dicts from propose_plans().
            parent_plan_dict: The parent plan dict (optional).

        Returns:
            De-duplicated list (preserves order, keeps first occurrence).
        """
        seen: set = set()

        if parent_plan_dict is not None:
            parent_sig = self._plan_signature(parent_plan_dict)
            seen.add(parent_sig)

        unique: List[Dict] = []
        for v in variants:
            sig = self._plan_signature(v)
            if sig not in seen:
                seen.add(sig)
                unique.append(v)
            else:
                self.logger.debug(
                    f"[Strategist] _deduplicate_variants: dropped duplicate "
                    f"variant with signature {sig}"
                )

        if len(unique) < len(variants):
            self.logger.info(
                f"[Strategist] _deduplicate_variants: removed "
                f"{len(variants) - len(unique)} duplicate variant(s)"
            )
        return unique

    @staticmethod
    def _mutate_params(variants: List[Dict]) -> List[Dict]:
        """Ensure numeric parameters differ across variants.

        For each numeric parameter that is identical across all variants,
        apply a small perturbation schedule so the MCTS tree explores different
        operating points.

        Perturbation schedules (applied to variant index i):
          sigma:       [0.5, 1.0, 1.5, 2.0, 2.5]
          min_size:    [32, 64, 128, 256, 512]
          filter_size: [3, 5, 7, 5, 3]
          iterations:  [1, 2, 3, 2, 1]
          kernel_size: [3, 5, 7, 5, 3]
        """
        _SCHEDULES: Dict[str, List] = {
            "sigma": [0.5, 1.0, 1.5, 2.0, 2.5],
            "min_size": [32, 64, 128, 256, 512],
            "filter_size": [3, 5, 7, 5, 3],
            "iterations": [1, 2, 3, 2, 1],
            "kernel_size": [3, 5, 7, 5, 3],
        }

        if len(variants) <= 1:
            return variants

        # Collect all param values per key across variants
        param_values: Dict[str, List] = {}
        for v in variants:
            for step in v.get("steps", []):
                for k, val in step.get("params", {}).items():
                    param_values.setdefault(k, []).append(val)

        # Find params that are identical across all variants
        homogeneous_keys: set = set()
        for k, vals in param_values.items():
            if k in _SCHEDULES and len(vals) >= len(variants):
                unique_vals = set(str(x) for x in vals)
                if len(unique_vals) == 1:
                    homogeneous_keys.add(k)

        if not homogeneous_keys:
            return variants

        # Apply schedule for each homogeneous param
        mutated = []
        for i, v in enumerate(variants):
            import copy

            v_copy = copy.deepcopy(v)
            for step in v_copy.get("steps", []):
                params = step.get("params", {})
                for k in list(params.keys()):
                    if k in homogeneous_keys and k in _SCHEDULES:
                        schedule = _SCHEDULES[k]
                        params[k] = schedule[i % len(schedule)]
            mutated.append(v_copy)

        return mutated

    # =====================================================================
    #  B. MCTS Expansion — propose candidate next actions
    # =====================================================================

    def run(
        self,
        state: MaskState,
        eval_result: Optional[EvalResult] = None,
        action_history: Optional[List[Action]] = None,
    ) -> List[Action]:
        """
        Propose candidate NEXT actions for MCTS expansion.

        Called at each MCTS iteration's Expansion step. The LLM reasons about
        the *current* mask state and the Evaluator's critique to suggest 2-3
        targeted next actions (not full plans — single tool calls).

        Args:
            state:          Current MaskState
            eval_result:    Evaluator's assessment of current state (None for root)
            action_history: Actions taken so far (for context)

        Returns:
            List of 2-3 candidate Actions, or [] if STOP is proposed
        """
        # 1. Build state description
        state_desc = self._build_state_description(state, action_history)

        # 2. Build eval feedback
        eval_feedback = self._build_eval_feedback(eval_result)

        # 3. Build memory priors
        memory_priors = self._build_memory_priors()

        # 4. Fill prompt template
        prompt = PROMPT_REGISTRY["STRATEGIST_EXPAND"].format(
            state_description=state_desc,
            toolbox_description=self._toolbox_desc,
            eval_feedback=eval_feedback,
            memory_priors=memory_priors,
        )

        system_prompt = (
            "You are a strategy planner for medical image segmentation. "
            "Propose candidate actions to improve pseudo-label quality. "
            "Respond ONLY with valid JSON."
        )

        try:
            response = self.llm_caller.call(
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=0.3,  # Slightly higher for diversity
            )
            actions = self._parse_response(response)
        except Exception as e:
            self.logger.error(f"[Strategist] MCTS expansion LLM call failed: {e}")
            actions = self._fallback_actions(eval_result)

        self.logger.info(
            f"[Strategist] case={state.case_id} step={state.step} "
            f"proposed {len(actions)} actions: "
            f"{[a.tool_id for a in actions]}"
        )
        return actions

    # =====================================================================
    #  Prior-based shortcut: convert plans → Action sequences
    # =====================================================================

    def get_prior_actions(self) -> List[List[Action]]:
        """
        Convert memory adaptation plans into Action sequences for MCTS Phase 0.5.

        Returns plans sorted by priority (plan_0 first = try first).
        Each plan's steps become a single Action sequence to be executed in order.
        """
        if self.memory is None:
            return []

        plans = self.memory.get_plans_by_priority()
        if plans:
            self.logger.info(
                f"[Strategist] get_prior_actions: using {len(plans)} adaptation plans "
                f"as prior action sequences"
            )
            return [plan.to_actions() for plan in plans]

        # Fallback to raw trajectory priors
        return self.memory.get_prior_actions()

    # =====================================================================
    #  Plan prompt construction helpers
    # =====================================================================

    def _format_domain_context(self, perceptor_plan: Dict) -> str:
        """Format the domain_context section of a perceptor plan for the prompt."""
        ctx = perceptor_plan.get("domain_context", {})
        if not ctx:
            return "Domain context: Unknown"

        lines = [
            f"Target organs: {', '.join(self.target_organs)}",
            f"Modality: {ctx.get('modality', 'Unknown')}",
            f"Anatomical site: {ctx.get('anatomical_site', 'Unknown')}",
            f"Contrast quality (1-10): {ctx.get('contrast_quality', 'Unknown')}",
            f"Noise assessment: {ctx.get('noise_assessment', 'Unknown')}",
            f"Artifacts: {', '.join(ctx.get('artifacts', ['None']))}",
        ]

        organ_pairs = ctx.get("organ_pairs", [])
        if organ_pairs:
            lines.append(f"Paired organs: {', '.join(str(p) for p in organ_pairs)}")

        small_targets = ctx.get("small_targets", [])
        if small_targets:
            names = [
                t["name"] if isinstance(t, dict) else str(t) for t in small_targets
            ]
            lines.append(f"Small targets (< 5% FOV): {', '.join(names)}")

        return "\n".join(lines)

    def _format_case_image_profile(self, case_profile: Optional[Dict]) -> str:
        """Format per-case image profile dict as text for the LLM prompt.

        This receives the flat dict returned by Perceptor.get_case_profile(),
        which contains statistics for a single case (not domain averages).
        """
        if not case_profile:
            return "(no per-case profile available — using domain averages only)"

        lines = ["Per-case statistics for this specific volume:"]

        def _add(key: str, label: str, unit: str = "") -> None:
            val = case_profile.get(key)
            if val is not None:
                lines.append(f"  {label}: {val}{unit}")

        _add("noise_std", "Noise std")
        _add("snr_estimate", "Estimated SNR")
        _add("foreground_ratio", "Foreground ratio")
        _add("intensity_skewness", "Intensity skewness")
        _add("intensity_p05", "5th percentile intensity")
        _add("intensity_p95", "95th percentile intensity")
        _add("histogram_num_peaks", "Histogram peaks")
        _add("likely_ct", "Likely CT")

        shape = case_profile.get("spatial_shape")
        if shape:
            lines.append(f"  Spatial shape: {shape}")

        # Add calibration hints derived from the stats
        noise_std = case_profile.get("noise_std")
        if noise_std is not None:
            sigma_hint = max(0.5, min(3.0, float(noise_std) / 15.0))
            lines.append(
                f"  [HINT] Recommended gaussian sigma ≈ {sigma_hint:.2f} "
                f"(calibrated from noise_std={noise_std})"
            )

        fg_ratio = case_profile.get("foreground_ratio")
        if fg_ratio is not None:
            if float(fg_ratio) < 0.01:
                min_size_hint = 32
            elif float(fg_ratio) < 0.05:
                min_size_hint = 64
            elif float(fg_ratio) < 0.15:
                min_size_hint = 128
            else:
                min_size_hint = 256
            lines.append(
                f"  [HINT] Recommended min_size ≈ {min_size_hint} "
                f"(calibrated from foreground_ratio={fg_ratio})"
            )

        return "\n".join(lines)

    def _format_image_profile(self, perceptor_plan: Dict) -> str:
        """Format the image_profile section for the prompt."""
        profile = perceptor_plan.get("image_profile", {})
        if not profile:
            return "(no programmatic profile available)"

        lines = []

        def _fmt_stat(key: str, label: str, unit: str = "") -> None:
            if key in profile:
                v = profile[key]
                if isinstance(v, dict):
                    lines.append(
                        f"- {label}: mean={v.get('mean', '?')}{unit}, "
                        f"range=[{v.get('min', '?')}, {v.get('max', '?')}]{unit}"
                    )
                else:
                    lines.append(f"- {label}: {v}{unit}")

        _fmt_stat("intensity_mean", "Mean intensity")
        _fmt_stat("intensity_std", "Intensity std")
        _fmt_stat("intensity_p05", "5th percentile intensity")
        _fmt_stat("intensity_p95", "95th percentile intensity")
        _fmt_stat("noise_std", "Noise std")
        _fmt_stat("snr_estimate", "Estimated SNR")
        _fmt_stat("foreground_ratio", "Foreground ratio")
        _fmt_stat("histogram_num_peaks", "Histogram peaks")

        if "likely_ct" in profile:
            lines.append(f"- Likely CT: {profile['likely_ct']}")
        if "noise_level_auto" in profile:
            lines.append(f"- Auto noise level: {profile['noise_level_auto']}")

        return "\n".join(lines) if lines else "(no profile statistics available)"

    # =====================================================================
    #  MCTS expansion prompt construction helpers
    # =====================================================================

    def _build_state_description(
        self,
        state: MaskState,
        action_history: Optional[List[Action]] = None,
    ) -> str:
        """Build text description of current state for LLM."""
        lines = [
            f"Case: {state.case_id}",
            f"Current step: {state.step}",
            f"Target organs: {', '.join(self.target_organs)}",
        ]

        # Geometric features
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

        # Action history
        if action_history:
            lines.append("")
            lines.append("Action history (most recent last):")
            for i, a in enumerate(action_history):
                lines.append(f"  Step {i + 1}: [{a.tool_id}] {a.description}")
        else:
            lines.append("No prior actions taken.")

        return "\n".join(lines)

    def _build_eval_feedback(self, eval_result: Optional[EvalResult]) -> str:
        """Build eval feedback text for LLM."""
        if eval_result is None:
            return "Initial state — no prior evaluation available."

        lines = [
            f"Reward: {eval_result.reward:.3f}",
            f"Decision: {eval_result.decision}",
        ]

        if eval_result.scores:
            lines.append("Dimension scores:")
            for dim, score in eval_result.scores.items():
                lines.append(f"  {dim}: {score}")

        if eval_result.reasoning:
            lines.append(f"Reasoning: {eval_result.reasoning}")

        if eval_result.suggested_actions:
            lines.append(f"Identified defects: {eval_result.suggested_actions}")

        return "\n".join(lines)

    def _build_memory_priors(self) -> str:
        """Build memory prior text for LLM (used in MCTS expansion prompt)."""
        if self.memory is None:
            return "No strategy memory available."

        priors = self.memory.get_prior_trajectories()
        if not priors:
            return "No prior experience available for this domain."

        lines = ["Top successful tool sequences from similar cases:"]
        for i, t in enumerate(priors):
            tools = [a["tool_id"] for a in t.actions]
            lines.append(
                f"  Path {i + 1}: {' → '.join(tools)} "
                f"(reward={t.reward:.2f}, used {t.count} times)"
            )
        return "\n".join(lines)

    # =====================================================================
    #  Plan response parsing
    # =====================================================================

    def _parse_plans_response(
        self, response: str, expected_num: int
    ) -> List[AdaptationPlan]:
        """Parse LLM JSON response for adaptation plans."""
        clean = self._clean_json_text(response)
        clean = self._sanitize_json_text(clean)
        clean = self._extract_json_object(clean)

        data = _loads_robust(clean)
        raw_plans = data.get("plans", [])

        plans = []
        seen_priorities = set()

        for raw in raw_plans:
            # Validate steps
            steps = raw.get("steps", [])
            valid_steps = []
            for step in steps:
                tool_id = step.get("tool_id", "").strip().lower()
                if tool_id in TOOL_REGISTRY:
                    valid_steps.append(
                        {"tool_id": tool_id, "params": step.get("params", {})}
                    )
                else:
                    self.logger.warning(
                        f"[Strategist] Unknown tool_id '{tool_id}' in plan "
                        f"'{raw.get('plan_id', '?')}', skipping step"
                    )

            if not valid_steps:
                self.logger.warning(
                    f"[Strategist] Plan '{raw.get('plan_id', '?')}' has no valid steps, skipping"
                )
                continue

            # Deduplicate priority
            priority = int(raw.get("priority", len(plans) + 1))
            while priority in seen_priorities:
                priority += 1
            seen_priorities.add(priority)

            plans.append(
                AdaptationPlan(
                    plan_id=raw.get("plan_id", f"plan_{len(plans)}"),
                    name=raw.get("name", f"Plan {len(plans)}"),
                    rationale=raw.get("rationale", ""),
                    steps=valid_steps,
                    priority=priority,
                    applicable_when=raw.get("applicable_when", ""),
                    expected_reward=float(raw.get("expected_reward", 0.7)),
                    source="llm_generated",
                )
            )

        # Log domain summary if available
        domain_summary = data.get("domain_summary", "")
        if domain_summary:
            self.logger.info(f"[Strategist] Domain summary: {domain_summary}")

        self.logger.info(
            f"[Strategist] Parsed {len(plans)}/{len(raw_plans)} valid plans"
        )
        return plans

    # =====================================================================
    #  MCTS expansion response parsing
    # =====================================================================

    def _parse_response(self, response: str) -> List[Action]:
        """Parse LLM JSON response into list of Actions (MCTS expansion)."""
        clean = self._clean_json_text(response)
        clean = self._sanitize_json_text(clean)
        clean = self._extract_json_object(clean)

        data = _loads_robust(clean)

        # Log CoT chain-of-thought from analysis field
        analysis = data.get("analysis", "")
        if analysis:
            self.logger.debug(f"[Strategist] expand CoT analysis: {analysis[:300]}")

        candidates = data.get("candidate_actions", [])
        actions = []

        for c in candidates:
            tool_id = c.get("tool_id", "").strip().lower()

            # Check for STOP signal
            if tool_id in ("stop", "none", ""):
                self.logger.info("[Strategist] STOP action proposed")
                continue

            # Validate tool_id against TOOL_REGISTRY
            if tool_id not in TOOL_REGISTRY:
                self.logger.warning(
                    f"[Strategist] Unknown tool_id '{tool_id}', skipping"
                )
                continue

            params = c.get("params", {})
            reasoning = c.get("reasoning", "")
            expected = c.get("expected_improvement", "")

            actions.append(
                Action(
                    tool_id=tool_id,
                    params=params,
                    description=f"{reasoning} | Expected: {expected}",
                )
            )

        return actions

    # =====================================================================
    #  Shared JSON utilities
    # =====================================================================

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Strip markdown code fences and whitespace."""
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:])
        clean = (
            clean.replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )
        return clean

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

    @staticmethod
    def _sanitize_json_text(text: str) -> str:
        """Pre-process LLM output to fix common JSON syntax issues.

        Handles:
        - JS-style single-line comments  (// ...)
        - JS-style block comments        (/* ... */)
        - Trailing commas before } or ]
        - Single-quoted strings → double-quoted
        - Unquoted object keys
        - Control characters inside strings
        """
        # 1. Remove single-line comments (// ...) but not inside strings
        #    Heuristic: lines where // appears outside a quoted context
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            # Simple heuristic: strip // comments that are NOT inside strings.
            # Count unescaped quotes before the // to decide.
            idx = line.find("//")
            if idx != -1:
                # Count unescaped double quotes before idx
                prefix = line[:idx]
                n_quotes = prefix.count('"') - prefix.count('\\"')
                if n_quotes % 2 == 0:
                    # // is outside a string → strip it
                    line = prefix.rstrip()
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # 2. Remove block comments /* ... */
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

        # 3. Remove control characters (except \n, \r, \t) that break JSON
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

        # 4. Trailing commas: ,  } or ,  ]
        text = re.sub(r",\s*([}\]])", r"\1", text)

        # 5. Single-quoted strings → double-quoted.
        #    Only convert if the text has no double-quoted strings at all
        #    (to avoid clobbering mixed-quote JSON).
        if '"' not in text and "'" in text:
            text = text.replace("'", '"')

        # 6. Unquoted keys: { key: or , key:
        #    Pattern: after { or , (with optional whitespace), a bare word
        #    followed by :  (not inside a string).
        text = re.sub(r"(?<=[{,])\s*([a-zA-Z_]\w*)\s*:", r' "\1":', text)

        return text

    def _fallback_plans(
        self, perceptor_plan: Dict, num_plans: int
    ) -> List[AdaptationPlan]:
        """
        Rule-based fallback plan generation when LLM is unavailable.
        Produces plans driven by programmatic image_profile statistics.
        """
        profile = perceptor_plan.get("image_profile", {})
        ctx = perceptor_plan.get("domain_context", {})
        noise_level = ctx.get("noise_assessment", "Medium")

        # Determine noise severity from profile if ctx is ambiguous
        snr_info = profile.get("snr_estimate", {})
        mean_snr = snr_info.get("mean", 30) if isinstance(snr_info, dict) else 30
        if mean_snr < 15:
            noise_level = "High"
        elif mean_snr < 30:
            noise_level = "Medium"

        contrast_quality = int(ctx.get("contrast_quality", 5))

        plans = []

        # Plan 1: Standard cleanup — always a safe baseline
        plans.append(
            AdaptationPlan(
                plan_id="plan_0",
                name="Standard Cleanup",
                rationale="Safe baseline: postprocess to fix topology errors",
                steps=[
                    {"tool_id": "biomedparse", "params": {}},
                    {
                        "tool_id": "postprocess_segmentation",
                        "params": {
                            "min_size": 100,
                            "keep_largest_only": True,
                            "fill_holes": True,
                        },
                    },
                ],
                priority=1,
                applicable_when="Any domain — conservative baseline",
                expected_reward=0.65,
                source="llm_generated",
            )
        )

        # Plan 2: Noise-aware
        if noise_level in ("High", "Medium"):
            smooth_params = (
                {"mode": "gaussian", "sigma": 1.5}
                if noise_level == "High"
                else {"mode": "gaussian", "sigma": 0.8}
            )
            plans.append(
                AdaptationPlan(
                    plan_id="plan_1",
                    name="Denoise + Segment + Cleanup",
                    rationale=f"Noise level={noise_level}: smooth first, then segment",
                    steps=[
                        {"tool_id": "smooth_image", "params": smooth_params},
                        {"tool_id": "biomedparse", "params": {}},
                        {
                            "tool_id": "postprocess_segmentation",
                            "params": {"keep_largest_only": True, "fill_holes": True},
                        },
                    ],
                    priority=2,
                    applicable_when=f"noise_level={noise_level}",
                    expected_reward=0.70,
                    source="llm_generated",
                )
            )

        # Plan 3: Low contrast → histogram equalization
        if contrast_quality < 5:
            plans.append(
                AdaptationPlan(
                    plan_id="plan_2",
                    name="Enhance Contrast + Segment",
                    rationale=f"Low contrast (quality={contrast_quality}): enhance first",
                    steps=[
                        {
                            "tool_id": "enhance_image",
                            "params": {"mode": "histogram_equalization"},
                        },
                        {"tool_id": "biomedparse", "params": {}},
                        {
                            "tool_id": "postprocess_segmentation",
                            "params": {"keep_largest_only": True, "fill_holes": True},
                        },
                    ],
                    priority=3,
                    applicable_when="contrast_quality < 5",
                    expected_reward=0.68,
                    source="llm_generated",
                )
            )

        # Plan 4: Dual-expert for severe domain shift
        plans.append(
            AdaptationPlan(
                plan_id="plan_3",
                name="Dual-Expert Fusion + Morphology",
                rationale="Combine BiomedParse + SAM3 for challenging domain shift",
                steps=[
                    {"tool_id": "dual_expert", "params": {}},
                    {"tool_id": "morphological_closing", "params": {"kernel_size": 3}},
                    {
                        "tool_id": "postprocess_segmentation",
                        "params": {"keep_largest_only": True, "fill_holes": True},
                    },
                ],
                priority=4,
                applicable_when="Severe domain shift / modality mismatch",
                expected_reward=0.72,
                source="llm_generated",
            )
        )

        self.logger.info(f"[Strategist] Fallback plans generated: {len(plans)} plans")
        return plans[:num_plans]

    def _fallback_actions(self, eval_result: Optional[EvalResult]) -> List[Action]:
        """
        Fallback when LLM is unavailable for MCTS expansion.
        Proposes sensible default actions based on eval feedback.
        """
        actions = []

        if eval_result is None:
            actions.append(
                Action(
                    tool_id="postprocess_segmentation",
                    params={
                        "min_size": 100,
                        "keep_largest_only": True,
                        "fill_holes": True,
                    },
                    description="Fallback: postprocess cleanup for initial mask",
                )
            )
            return actions

        defects = eval_result.suggested_actions or []
        defect_text = " ".join(str(d) for d in defects).lower()

        if any(k in defect_text for k in ("fragment", "component", "noise")):
            actions.append(
                Action(
                    tool_id="postprocess_segmentation",
                    params={
                        "min_size": 100,
                        "keep_largest_only": True,
                        "fill_holes": False,
                    },
                    description="Fallback: remove fragments",
                )
            )

        if any(k in defect_text for k in ("hole", "gap")):
            actions.append(
                Action(
                    tool_id="morphological_closing",
                    params={"kernel_size": 3, "iterations": 1},
                    description="Fallback: fill holes via morphological closing",
                )
            )

        if any(k in defect_text for k in ("boundary", "jagged", "smooth")):
            actions.append(
                Action(
                    tool_id="gaussian_smooth_mask",
                    params={"sigma": 1.0},
                    description="Fallback: smooth boundaries",
                )
            )

        if any(k in defect_text for k in ("missing", "small")):
            actions.append(
                Action(
                    tool_id="morphological_closing",
                    params={"kernel_size": 3, "iterations": 1},
                    description="Fallback: morphological closing for missing regions",
                )
            )

        if not actions:
            actions.append(
                Action(
                    tool_id="postprocess_segmentation",
                    params={
                        "min_size": 100,
                        "keep_largest_only": True,
                        "fill_holes": True,
                    },
                    description="Fallback: default postprocess cleanup",
                )
            )

        return actions[:3]  # Max 3 candidates

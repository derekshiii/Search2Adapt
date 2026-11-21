"""
Search2Adapt Dynamic Strategy Prior Memory
动态策略先验记忆库

设计动机:
  避免每个样本都从零开始搜索, 将成功的工具组合路径 (Trajectory)
  写入 JSON 记忆库, 面对新样本时优先尝试高频成功路径

统一记忆文件结构 (memory_<domain_key>.json):
  {
    "domain_key":  "AMOS-CT_SFUDA",
    "created_at":  "2026-03-01T10:00:00",
    "updated_at":  "2026-03-01T10:05:00",

    "domain_profile": {          ← 由 Perceptor 写入 (一次性)
      "text_prompts":    {...},
      "organ_label_map": {...},
      "domain_context":  {...},
      "image_profile":   {...},
      "data_info":       {...}
    },

    "adaptation_plans": [...],   ← 由 Strategist 写入 (一次性)

    "trajectories": [            ← 由 MCTS 增量写入 (per-case)
      {
        "actions":  [{"tool_id": "clahe", "params": {...}}, ...],
        "reward":   0.92,
        "count":    15,
        "case_ids": ["case_001", ...]
      }
    ],

    "eval_records": [            ← 由 Evaluator 增量写入 (per-step)
      {
        "case_id":   "AMOS_CT_0001",
        "step":      2,
        "mode":      "visual",
        "timestamp": "2026-03-01T21:55:00",
        "reward":    0.73,
        "scores": {
          "anatomical_plausibility": 0.80,
          "boundary_precision": 0.60,
          "morphological_consistency": 0.60,
          "false_positives": 0.80
        },
        "decision":    "CONTINUE",
        "defects":     ["jagged boundaries on liver"],
        "reasoning":   "Segmentation covers correct regions but liver boundary is rough",
        "grid_path":   "./S2A_Results/.../eval_grids/AMOS_CT_0001_step2_grid.png",
        "action_taken": "gaussian_smooth_mask"
      }
    ]
  }
"""

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    # Only for type checkers — not executed at runtime, breaks circular import
    from search2adapt.agents.base import Action


def _params_key(params: Optional[Dict]) -> tuple:
    """Hashable, order-independent key for tool params.

    Param values are not always scalars — e.g. ``label_filter`` takes
    ``applied_labels=[0, 1]`` — so ``tuple(sorted(params.items()))`` raises
    ``TypeError: unhashable type: 'list'``. Serialise every value instead.
    """
    return tuple(
        sorted(
            (str(key), json.dumps(value, sort_keys=True, default=str))
            for key, value in (params or {}).items()
        )
    )


@dataclass
class AdaptationPlan:
    """
    一条完整的 Adaptation Plan — 由 Strategist 基于 Perceptor 感知结果生成.

    每个 plan 是一条完整的工具链 (tool-chain), 描述了从粗标签到精细化伪标签的
    完整执行路径. 多个 plans 构成候选集, MCTS 从中选择并评估.

    字段:
      plan_id:        唯一标识 (e.g. "plan_0")
      name:           简短可读名称 (e.g. "Denoising + BiomedParse + Cleanup")
      rationale:      LLM 给出的选择该 plan 的理由 (对应 domain 感知结果)
      steps:          有序工具步骤列表, 每步包含 tool_id + params
      priority:       执行优先级 (1=最高), 决定 MCTS 先验尝试顺序
      applicable_when: 适用的 domain 特征描述 (用于向 Strategist 解释)
      expected_reward: LLM 估计的预期 reward
      source:         "llm_generated" | "memory_promoted" (从 trajectory 提升而来)
    """

    plan_id: str
    name: str
    rationale: str
    steps: List[Dict]  # [{"tool_id": ..., "params": {...}}, ...]
    priority: int = 1
    applicable_when: str = ""
    expected_reward: float = 0.7
    source: str = "llm_generated"

    def to_dict(self) -> Dict:
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "rationale": self.rationale,
            "steps": self.steps,
            "priority": self.priority,
            "applicable_when": self.applicable_when,
            "expected_reward": self.expected_reward,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AdaptationPlan":
        return cls(
            plan_id=d["plan_id"],
            name=d["name"],
            rationale=d.get("rationale", ""),
            steps=d["steps"],
            priority=d.get("priority", 1),
            applicable_when=d.get("applicable_when", ""),
            expected_reward=d.get("expected_reward", 0.7),
            source=d.get("source", "llm_generated"),
        )

    def to_actions(self) -> "List[Action]":
        """Convert steps to Action objects for MCTS execution."""
        from search2adapt.agents.base import Action  # lazy import, breaks circular dep

        return [
            Action(
                tool_id=step["tool_id"],
                params=step.get("params", {}),
                description=f"[Plan:{self.plan_id}] {self.name}",
            )
            for step in self.steps
        ]


@dataclass
class TrajectoryRecord:
    """一条工具链路径的记录"""

    actions: List[Dict]  # [{"tool_id": ..., "params": ...}, ...]
    reward: float  # 最佳 reward
    count: int = 1  # 成功使用次数
    case_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "actions": self.actions,
            "reward": self.reward,
            "count": self.count,
            "case_ids": self.case_ids,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TrajectoryRecord":
        return cls(
            actions=d["actions"],
            reward=d["reward"],
            count=d.get("count", 1),
            case_ids=d.get("case_ids", []),
        )


@dataclass
class EvalRecord:
    """
    单次 Evaluator 评估记录 — 由 Evaluator agent 写入

    每次对某个 case 的某个 step 进行评估后, 追加一条记录到 eval_records 列表.
    这些记录构成 MCTS 的评估历史, 供 Strategist 查阅以决定后续搜索方向.

    字段:
      case_id:      样本 ID (e.g. "AMOS_CT_0001")
      step:         当前 MCTS 步数 (对应 MaskState.step)
      mode:         评估模式: "text" | "visual"
      timestamp:    ISO 时间戳
      reward:       归一化 reward ∈ [0, 1]
      scores:       各维度得分 (1-5 整数)
      decision:     "ACCEPT" | "CONTINUE" | "REJECT"
      defects:      识别出的缺陷列表
      reasoning:    LLM/VLM 给出的推理说明
      grid_path:    overlay grid PNG 路径 (visual 模式时非空)
      action_taken: 触发本次评估的工具 ID (空串表示初始评估)
    """

    case_id: str
    step: int
    mode: str  # "text" | "visual"
    timestamp: str
    reward: float
    scores: Dict
    decision: str
    defects: List[str] = field(default_factory=list)
    reasoning: str = ""
    grid_path: str = ""
    action_taken: str = ""

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "step": self.step,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "reward": self.reward,
            "scores": self.scores,
            "decision": self.decision,
            "defects": self.defects,
            "reasoning": self.reasoning,
            "grid_path": self.grid_path,
            "action_taken": self.action_taken,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "EvalRecord":
        return cls(
            case_id=d["case_id"],
            step=d["step"],
            mode=d.get("mode", "text"),
            timestamp=d.get("timestamp", ""),
            reward=d["reward"],
            scores=d.get("scores", {}),
            decision=d.get("decision", "CONTINUE"),
            defects=d.get("defects", []),
            reasoning=d.get("reasoning", ""),
            grid_path=d.get("grid_path", ""),
            action_taken=d.get("action_taken", ""),
        )


@dataclass
class PlanDialogueRecord:
    """
    CoTS Plan Dialogue Record — 存储成功的计划 + 对话 (由 MCTS 写入)

    对应 CoTS 论文中 "extended memory" 概念:
      中期存储协作计划, 用于跨 case 复用成功的计划模式.

    字段:
      plan_steps:          完整工具链 (序列化)
      strategist_proposal: Strategist 的计划提案描述
      evaluator_critique:  Evaluator 的评估与建议
      plan_reward:         计划级 reward (LLM 评估)
      execution_reward:    实际执行 mask 质量 reward
      case_ids:            适用的样本 ID 列表
      count:               被成功使用的次数
    """

    plan_steps: List[Dict]  # [{"tool_id": ..., "params": ...}, ...]
    strategist_proposal: str = ""
    evaluator_critique: str = ""
    plan_reward: float = 0.0
    execution_reward: float = 0.0
    case_ids: List[str] = field(default_factory=list)
    count: int = 1

    def to_dict(self) -> Dict:
        return {
            "plan_steps": self.plan_steps,
            "strategist_proposal": self.strategist_proposal,
            "evaluator_critique": self.evaluator_critique,
            "plan_reward": self.plan_reward,
            "execution_reward": self.execution_reward,
            "case_ids": self.case_ids,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PlanDialogueRecord":
        return cls(
            plan_steps=d.get("plan_steps", []),
            strategist_proposal=d.get("strategist_proposal", ""),
            evaluator_critique=d.get("evaluator_critique", ""),
            plan_reward=d.get("plan_reward", 0.0),
            execution_reward=d.get("execution_reward", 0.0),
            case_ids=d.get("case_ids", []),
            count=d.get("count", 1),
        )


class StrategyMemory:
    """
    动态策略先验记忆库 — 统一单文件设计

    所有数据写入同一个 JSON 文件: memory_<domain_key>.json

    数据层:
      - domain_profile: Perceptor 感知结果 (一次性)
      - adaptation_plans: Strategist 生成的候选计划 (一次性)
      - trajectories: MCTS 增量写入的成功路径
      - eval_records: Evaluator 增量写入的评估记录
      - plan_dialogues: CoTS 增量写入的计划对话记录 (extended memory)

    Prior 机制:
      面对同一 Target Domain 的新样本时, Strategist 首先提取高频成功路径
      作为"强先验 (Strong Prior)", 如果该路径 Evaluation 及格, 则直接采纳
    """

    def __init__(
        self,
        memory_dir: str,
        domain_key: str = "default",
        max_trajectories: int = 500,
        prior_top_k: int = 5,
        prior_min_reward: float = 0.7,
        prior_accept_threshold: float = 0.75,
        logger: Optional[logging.Logger] = None,
    ):
        self.memory_dir = memory_dir
        self.domain_key = domain_key
        self.max_trajectories = max_trajectories
        self.prior_top_k = prior_top_k
        self.prior_min_reward = prior_min_reward
        self.prior_accept_threshold = prior_accept_threshold
        self.logger = logger or logging.getLogger("StrategyMemory")

        os.makedirs(memory_dir, exist_ok=True)
        # Single unified file — replaces both memory_<>.json and
        # adaptation_plans_<>.json from the old design
        self.unified_file = os.path.join(memory_dir, f"memory_{domain_key}.json")

        # In-memory state
        self.domain_profile: Optional[Dict] = None
        self.trajectories: List[TrajectoryRecord] = []
        self.adaptation_plans: List[AdaptationPlan] = []
        self.eval_records: List[EvalRecord] = []
        self.plan_dialogues: List[PlanDialogueRecord] = []  # CoTS extended memory

        # Timestamps
        self._created_at: Optional[str] = None

        # Dirty flag — avoids O(N²) disk writes on every add_* call.
        # Set True whenever in-memory state changes; flush() writes to disk.
        self._dirty: bool = False

        # O(1) fingerprint index for trajectory deduplication.
        # Maps fingerprint tuple → index in self.trajectories.
        self._trajectory_fingerprints: dict = {}

        # Thread-safety: all mutating operations acquire this lock.
        self._lock = threading.Lock()

        self._load()

    # =====================================================================
    #  Load / Save — unified
    # =====================================================================

    def _load(self):
        """Load all sections from the unified JSON file."""
        if os.path.exists(self.unified_file):
            try:
                with open(self.unified_file, "r") as f:
                    data = json.load(f)

                self._created_at = data.get("created_at")
                self.domain_profile = data.get("domain_profile", None)
                self.trajectories = [
                    TrajectoryRecord.from_dict(t) for t in data.get("trajectories", [])
                ]
                self.adaptation_plans = [
                    AdaptationPlan.from_dict(p)
                    for p in data.get("adaptation_plans", [])
                ]
                self.eval_records = [
                    EvalRecord.from_dict(e) for e in data.get("eval_records", [])
                ]
                self.plan_dialogues = [
                    PlanDialogueRecord.from_dict(pd)
                    for pd in data.get("plan_dialogues", [])
                ]
                self.logger.info(
                    f"[Memory] Loaded unified memory: "
                    f"domain_profile={'yes' if self.domain_profile else 'no'}, "
                    f"{len(self.adaptation_plans)} plans, "
                    f"{len(self.trajectories)} trajectories, "
                    f"{len(self.eval_records)} eval_records, "
                    f"{len(self.plan_dialogues)} plan_dialogues"
                )
            except Exception as e:
                self.logger.warning(f"[Memory] Failed to load unified memory: {e}")
                self.domain_profile = None
                self.trajectories = []
                self.adaptation_plans = []
                self.plan_dialogues = []
        else:
            self.logger.info("[Memory] No existing memory found, starting fresh")

        # Rebuild O(1) fingerprint index after loading
        self._rebuild_fingerprint_index()

    def _rebuild_fingerprint_index(self):
        """Rebuild the O(1) trajectory fingerprint index from self.trajectories."""
        self._trajectory_fingerprints = {}
        for idx, t in enumerate(self.trajectories):
            fp = tuple(
                (a["tool_id"], _params_key(a.get("params")))
                for a in t.actions
            )
            self._trajectory_fingerprints[fp] = idx

    def _mark_dirty(self):
        """Mark in-memory state as modified (needs flush to disk)."""
        self._dirty = True

    def flush(self):
        """Write to disk only if there are pending changes."""
        with self._lock:
            if self._dirty:
                self.save()

    def save(self):
        """Persist all sections to the unified JSON file (atomic write)."""
        now = datetime.now().isoformat()
        data = {
            "domain_key": self.domain_key,
            "created_at": self._created_at or now,
            "updated_at": now,
            "domain_profile": self.domain_profile,
            "adaptation_plans": [p.to_dict() for p in self.adaptation_plans],
            "trajectories": [t.to_dict() for t in self.trajectories],
            "eval_records": [e.to_dict() for e in self.eval_records],
            "plan_dialogues": [pd.to_dict() for pd in self.plan_dialogues],
        }
        # Atomic write: write to temp file then rename to avoid corruption
        # on crash or concurrent reads.
        dir_name = os.path.dirname(self.unified_file)
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=dir_name, prefix=".memory_", suffix=".tmp"
            )
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.unified_file)
        except OSError:
            # Fallback: direct write (e.g. if rename fails across filesystems)
            with open(self.unified_file, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        if self._created_at is None:
            self._created_at = now
        self._dirty = False
        self.logger.info(
            f"[Memory] Saved unified memory: "
            f"{len(self.adaptation_plans)} plans, "
            f"{len(self.trajectories)} trajectories, "
            f"{len(self.eval_records)} eval_records, "
            f"{len(self.plan_dialogues)} plan_dialogues → {self.unified_file}"
        )

    def __del__(self):
        """Flush pending changes on garbage collection."""
        try:
            self.flush()
        except Exception:
            pass

    # =====================================================================
    #  Domain Profile — written once by Perceptor
    # =====================================================================

    def save_domain_profile(self, profile: Dict):
        """
        Store the full Perceptor output as domain_profile and persist.
        Should be called once at the end of Perceptor.run().
        """
        with self._lock:
            self.domain_profile = profile
            self.save()
        self.logger.info("[Memory] Domain profile saved to unified memory")

    def get_domain_profile(self) -> Optional[Dict]:
        """Return the stored domain profile (None if not yet set)."""
        return self.domain_profile

    # =====================================================================
    #  Adaptation Plans — written once by Strategist
    # =====================================================================

    def set_adaptation_plans(self, plans: List[AdaptationPlan]):
        """
        Replace the current plan set with a freshly generated list.
        Called by Strategist.generate_adaptation_plans() after LLM generation.
        Also promotes top memory trajectories as additional plans.
        """
        with self._lock:
            self.adaptation_plans = plans
            # Promote top successful trajectories into plans (source = memory_promoted)
            promoted = self._promote_trajectories_to_plans(
                existing_ids={p.plan_id for p in plans}
            )
            self.adaptation_plans.extend(promoted)
            # Sort by priority (ascending = highest priority first)
            self.adaptation_plans.sort(key=lambda p: p.priority)
            self.save()
        self.logger.info(
            f"[Memory] Set {len(plans)} LLM plans + "
            f"{len(promoted)} promoted trajectory plans"
        )

    def add_plan(self, plan: AdaptationPlan):
        """Add or update a single plan (upsert by plan_id)."""
        with self._lock:
            for i, existing in enumerate(self.adaptation_plans):
                if existing.plan_id == plan.plan_id:
                    self.adaptation_plans[i] = plan
                    self.save()
                    return
            self.adaptation_plans.append(plan)
            self.adaptation_plans.sort(key=lambda p: p.priority)
            self.save()

    def get_plans_by_priority(self) -> List[AdaptationPlan]:
        """Return all plans sorted by priority (1 = highest)."""
        return sorted(self.adaptation_plans, key=lambda p: p.priority)

    def get_plan_by_id(self, plan_id: str) -> Optional[AdaptationPlan]:
        """Look up a plan by its ID."""
        for p in self.adaptation_plans:
            if p.plan_id == plan_id:
                return p
        return None

    def _promote_trajectories_to_plans(self, existing_ids: set) -> List[AdaptationPlan]:
        """
        Convert top-K high-reward MCTS trajectories into AdaptationPlan objects
        so the Strategist can try them as prior plans in future runs.
        """
        promoted = []
        top_trajs = self.get_prior_trajectories()
        for i, traj in enumerate(top_trajs):
            plan_id = f"memory_plan_{i}"
            if plan_id in existing_ids:
                continue
            promoted.append(
                AdaptationPlan(
                    plan_id=plan_id,
                    name=" → ".join(a["tool_id"] for a in traj.actions),
                    rationale=(
                        f"Promoted from MCTS memory: used {traj.count} times, "
                        f"best reward={traj.reward:.3f}"
                    ),
                    steps=[
                        {"tool_id": a["tool_id"], "params": a.get("params", {})}
                        for a in traj.actions
                    ],
                    priority=len(existing_ids) + i + 1,  # Lower priority than LLM plans
                    applicable_when="High-reward MCTS trajectory from prior cases",
                    expected_reward=traj.reward,
                    source="memory_promoted",
                )
            )
        return promoted

    # =====================================================================
    #  Trajectories — written incrementally by MCTS
    # =====================================================================

    def add_trajectory(self, actions: "List[Action]", reward: float, case_id: str):
        """
        将成功路径写入记忆库
        如果已有相同路径, 增加 count; 否则新增
        """
        if reward < self.prior_min_reward:
            return  # 低于阈值不写入

        with self._lock:
            action_dicts = [a.to_dict() for a in actions]
            # Use (tool_id, sorted params) as fingerprint so that the same tool
            # with different params (e.g. gaussian_smooth_mask(sigma=0.5) vs
            # sigma=3.0) is treated as a different trajectory.
            fingerprint = tuple(
                (a.tool_id, _params_key(a.params))
                for a in actions
            )

            # O(1) lookup via fingerprint index
            existing_idx = self._trajectory_fingerprints.get(fingerprint)
            if existing_idx is not None and existing_idx < len(self.trajectories):
                t = self.trajectories[existing_idx]
                # 已有相同路径, 更新
                t.count += 1
                t.reward = max(t.reward, reward)  # 保留最优 reward
                if case_id not in t.case_ids:
                    t.case_ids.append(case_id)
                self.save()
                return

            # 新路径
            record = TrajectoryRecord(
                actions=action_dicts,
                reward=reward,
                count=1,
                case_ids=[case_id],
            )
            new_idx = len(self.trajectories)
            self.trajectories.append(record)
            self._trajectory_fingerprints[fingerprint] = new_idx

            # 超出上限时淘汰 reward 最低的
            if len(self.trajectories) > self.max_trajectories:
                self.trajectories.sort(key=lambda t: t.reward, reverse=True)
                self.trajectories = self.trajectories[: self.max_trajectories]
                # Rebuild index after eviction (order changed)
                self._rebuild_fingerprint_index()

            self.save()

    def get_prior_trajectories(self) -> List[TrajectoryRecord]:
        """
        获取强先验路径 — 按 count * reward 排序, 取 top_k
        """
        if not self.trajectories:
            return []

        scored = sorted(
            self.trajectories,
            key=lambda t: t.count * t.reward,
            reverse=True,
        )
        return scored[: self.prior_top_k]

    def get_prior_actions(self) -> "List[List[Action]]":
        """
        获取先验路径, 转换为 Action 列表
        """
        from search2adapt.agents.base import Action  # lazy import, breaks circular dep

        priors = self.get_prior_trajectories()
        result = []
        for t in priors:
            actions = [Action.from_dict(a) for a in t.actions]
            result.append(actions)
        return result

    # =====================================================================
    #  Eval Records — written incrementally by Evaluator
    # =====================================================================

    def add_eval_record(self, record: "EvalRecord"):
        """
        Append a single evaluation record and persist to the unified file.
        Called by Evaluator after each _text_eval() or _visual_eval().
        """
        with self._lock:
            self.eval_records.append(record)
            self.save()
        self.logger.debug(
            f"[Memory] EvalRecord added: case={record.case_id} "
            f"step={record.step} mode={record.mode} reward={record.reward:.3f}"
        )

    def get_eval_records(
        self,
        case_id: Optional[str] = None,
        mode: Optional[str] = None,
        min_reward: float = 0.0,
    ) -> "List[EvalRecord]":
        """
        Return filtered eval records.

        Args:
            case_id:    if given, filter to this case only
            mode:       if given, filter to "text" or "visual"
            min_reward: only return records with reward >= this threshold
        """
        records = self.eval_records
        if case_id is not None:
            records = [r for r in records if r.case_id == case_id]
        if mode is not None:
            records = [r for r in records if r.mode == mode]
        if min_reward > 0.0:
            records = [r for r in records if r.reward >= min_reward]
        return records

    def get_case_eval_history(self, case_id: str) -> "List[EvalRecord]":
        """
        Return all evaluation records for a specific case, sorted by step.
        Useful for MCTS to review per-case progress history.
        """
        return sorted(
            [r for r in self.eval_records if r.case_id == case_id],
            key=lambda r: r.step,
        )

    def get_eval_stats(self) -> Dict:
        """Evaluation record summary statistics."""
        if not self.eval_records:
            return {"total": 0, "avg_reward": 0.0, "by_mode": {}, "by_decision": {}}

        total = len(self.eval_records)
        avg_reward = sum(r.reward for r in self.eval_records) / total

        by_mode: Dict[str, int] = {}
        by_decision: Dict[str, int] = {}
        for r in self.eval_records:
            by_mode[r.mode] = by_mode.get(r.mode, 0) + 1
            by_decision[r.decision] = by_decision.get(r.decision, 0) + 1

        return {
            "total": total,
            "avg_reward": round(avg_reward, 4),
            "by_mode": by_mode,
            "by_decision": by_decision,
        }

    def get_stats(self) -> Dict:
        """记忆库统计"""
        if not self.trajectories:
            return {"total": 0, "avg_reward": 0, "top_paths": []}

        return {
            "total": len(self.trajectories),
            "avg_reward": sum(t.reward for t in self.trajectories)
            / len(self.trajectories),
            "top_paths": [
                {
                    "tools": [a["tool_id"] for a in t.actions],
                    "reward": t.reward,
                    "count": t.count,
                }
                for t in self.get_prior_trajectories()
            ],
        }

    # =====================================================================
    #  Plan Dialogues — CoTS extended memory (written by MCTS)
    # =====================================================================

    def add_plan_dialogue(
        self,
        plan_steps: "List[Action]",
        strategist_proposal: str,
        evaluator_critique: str,
        plan_reward: float,
        execution_reward: float,
        case_id: str,
    ):
        """
        将成功的计划对话写入 extended memory (CoTS 风格)

        对应 CoTS 论文的 extended memory: 中期存储协作计划,
        平衡短期和长期记忆, 用于跨 case 复用成功的计划模式.

        Args:
            plan_steps:          执行成功的完整工具链
            strategist_proposal: Strategist 的计划提案
            evaluator_critique:  Evaluator 的评估
            plan_reward:         LLM 评估的计划 reward
            execution_reward:    实际执行的 mask reward
            case_id:             样本 ID
        """
        from search2adapt.agents.base import Action  # lazy import

        step_dicts = [s.to_dict() if isinstance(s, Action) else s for s in plan_steps]

        # 指纹匹配: 如果已有相同工具链, 更新计数和 reward
        fingerprint = tuple(
            (s["tool_id"], _params_key(s.get("params")))
            for s in step_dicts
        )

        with self._lock:
            for pd in self.plan_dialogues:
                existing_fp = tuple(
                    (s["tool_id"], _params_key(s.get("params")))
                    for s in pd.plan_steps
                )
                if existing_fp == fingerprint:
                    pd.count += 1
                    pd.execution_reward = max(pd.execution_reward, execution_reward)
                    pd.plan_reward = max(pd.plan_reward, plan_reward)
                    if case_id not in pd.case_ids:
                        pd.case_ids.append(case_id)
                    self.save()
                    return

            # 新记录
            record = PlanDialogueRecord(
                plan_steps=step_dicts,
                strategist_proposal=strategist_proposal,
                evaluator_critique=evaluator_critique,
                plan_reward=plan_reward,
                execution_reward=execution_reward,
                case_ids=[case_id],
                count=1,
            )
            self.plan_dialogues.append(record)
            self.save()
        self.logger.info(
            f"[Memory] Plan dialogue added: "
            f"steps={[s['tool_id'] for s in step_dicts]}, "
            f"plan_reward={plan_reward:.3f}, exec_reward={execution_reward:.3f}"
        )

    def get_plan_dialogues(
        self,
        top_k: int = 5,
        min_execution_reward: float = 0.0,
    ) -> List[PlanDialogueRecord]:
        """
        获取成功的计划对话记录, 按 count * execution_reward 排序

        Args:
            top_k:                返回前 K 条
            min_execution_reward: 最低执行 reward 过滤
        """
        filtered = [
            pd
            for pd in self.plan_dialogues
            if pd.execution_reward >= min_execution_reward
        ]
        scored = sorted(
            filtered,
            key=lambda pd: pd.count * pd.execution_reward,
            reverse=True,
        )
        return scored[:top_k]

    def get_plan_dialogue_summaries(self, top_k: int = 3) -> List[str]:
        """
        获取成功计划对话的文本摘要, 用于注入 Strategist 提示词

        Returns:
            List of human-readable plan dialogue summaries
        """
        dialogues = self.get_plan_dialogues(top_k=top_k)
        summaries = []
        for i, pd in enumerate(dialogues):
            tools = " → ".join(s["tool_id"] for s in pd.plan_steps)
            summary = (
                f"Plan {i + 1} (used {pd.count}x, "
                f"exec_reward={pd.execution_reward:.3f}):\n"
                f"  Tools: {tools}\n"
                f"  Proposal: {pd.strategist_proposal[:200]}...\n"
                f"  Critique: {pd.evaluator_critique[:200]}..."
            )
            summaries.append(summary)
        return summaries

    # =====================================================================
    #  Plan Eval Records — Phase 1 & 2 evaluation persistence (Steps 1-3)
    # =====================================================================

    def add_plan_eval_record(
        self,
        case_id: str,
        step: int,
        mode: str,  # "plan_eval" | "viability"
        reward: float,
        decision: str,
        defects: List[str],
        reasoning: str,
        action_taken: str = "",
        scores: Optional[Dict] = None,
    ):
        """
        将 plan_eval / viability_eval 结果持久化到 eval_records.

        与 add_eval_record 复用同一 EvalRecord 结构,
        用 mode="plan_eval" 或 mode="viability" 区分.

        Args:
            case_id:      样本 ID
            step:         当前 MCTS plan iteration 编号
            mode:         "plan_eval" | "viability"
            reward:       plan reward 或 viability reward
            decision:     "ACCEPT" | "CONTINUE" | "REJECT" | "VIABLE" | "ABORT"
            defects:      identified defects / suggestions list
            reasoning:    LLM critique / reasoning text
            action_taken: plan step description (e.g. tool chain string)
            scores:       dimension scores dict (optional)
        """
        record = EvalRecord(
            case_id=case_id,
            step=step,
            mode=mode,
            timestamp=datetime.now().isoformat(),
            reward=reward,
            scores=scores or {},
            decision=decision,
            defects=defects,
            reasoning=reasoning,
            grid_path="",
            action_taken=action_taken,
        )
        with self._lock:
            self.eval_records.append(record)
            self.save()
        self.logger.debug(
            f"[Memory] PlanEvalRecord added: case={case_id} "
            f"step={step} mode={mode} reward={reward:.3f}"
        )

    # =====================================================================
    #  Domain Context Summary — Step 2
    # =====================================================================

    def get_all_eval_records(self) -> List["EvalRecord"]:
        """
        Return a shallow copy of all evaluation records stored in memory.

        Useful for cross-case analysis such as reward credibility reporting
        (correlation between LLM reward and GT Dice).

        Returns:
            List of EvalRecord objects in insertion order.
        """
        return list(self.eval_records)

    def get_domain_context_summary(self) -> str:
        """
        将 domain_profile 格式化为 agent 可读的文本摘要.

        读取 Perceptor 写入的 domain_profile (包含 domain_context,
        image_profile, text_prompts, organ_label_map), 生成一段结构化
        文字供 Strategist / Evaluator 的 prompt 使用.

        Returns:
            Multi-line string with key domain characteristics,
            or a placeholder string if domain_profile is not yet set.
        """
        if self.domain_profile is None:
            return "(Domain profile not yet available — Perceptor has not run)"

        ctx = self.domain_profile.get("domain_context", {})
        profile = self.domain_profile.get("image_profile", {})
        organ_label_map = self.domain_profile.get("organ_label_map", {})
        text_prompts = self.domain_profile.get("text_prompts", {})

        lines = ["=== Domain Profile Summary ==="]

        # --- Core domain context ---
        modality = ctx.get("modality", "Unknown")
        site = ctx.get("anatomical_site", "Unknown")
        contrast = ctx.get("contrast_quality", "Unknown")
        noise = ctx.get("noise_assessment", "Unknown")
        artifacts = ctx.get("artifacts", ["None"])
        lines.append(f"Modality: {modality}")
        lines.append(f"Anatomical site: {site}")
        lines.append(f"Contrast quality (1-10): {contrast}")
        lines.append(f"Noise level: {noise}")
        lines.append(f"Artifacts: {', '.join(str(a) for a in artifacts)}")

        # --- Target organs ---
        organs = list(organ_label_map.keys()) or list(text_prompts.keys())
        if organs:
            lines.append(f"Target organs: {', '.join(organs)}")
            lines.append(f"Label map: {organ_label_map}")

        # --- Paired organs and small targets ---
        organ_pairs = ctx.get("organ_pairs", [])
        if organ_pairs:
            pair_strs = [
                f"{p[0]} & {p[1]}"
                if isinstance(p, (list, tuple)) and len(p) == 2
                else str(p)
                for p in organ_pairs
            ]
            lines.append(f"Paired organs: {', '.join(pair_strs)}")

        small_targets = ctx.get("small_targets", [])
        if small_targets:
            names = [
                t["name"] if isinstance(t, dict) else str(t) for t in small_targets
            ]
            lines.append(f"Small targets (< 5% FOV): {', '.join(names)}")

        # --- Image profile statistics (key metrics only) ---
        if profile:
            lines.append("--- Image Statistics ---")
            snr = profile.get("snr_estimate", {})
            if isinstance(snr, dict) and "mean" in snr:
                lines.append(f"SNR (mean): {snr['mean']:.1f}")
            elif isinstance(snr, (int, float)):
                lines.append(f"SNR: {snr:.1f}")

            fg = profile.get("foreground_ratio", {})
            if isinstance(fg, dict) and "mean" in fg:
                lines.append(f"Foreground ratio (mean): {fg['mean']:.3f}")
            elif isinstance(fg, (int, float)):
                lines.append(f"Foreground ratio: {fg:.3f}")

            noise_auto = profile.get("noise_level_auto", "")
            if noise_auto:
                lines.append(f"Auto noise classification: {noise_auto}")

            likely_ct = profile.get("likely_ct", None)
            if likely_ct is not None:
                lines.append(f"Likely CT: {likely_ct}")

        return "\n".join(lines)

    # =====================================================================
    #  Case Search Context — Step 3
    # =====================================================================

    def get_case_search_context(
        self,
        case_id: str,
        max_eval_records: int = 10,
        max_plan_dialogues: int = 3,
    ) -> str:
        """
        为特定 case 生成完整的搜索上下文摘要, 供 agent prompt 注入.

        汇总:
          1. 该 case 的全部评估历史 (text/visual/plan_eval/viability)
          2. 全局 plan dialogue 记忆 (top-K 成功计划)

        Args:
            case_id:           样本 ID
            max_eval_records:  最多包含最近 N 条评估记录
            max_plan_dialogues: 最多包含 top-K 计划对话

        Returns:
            Formatted multi-line string for prompt injection.
        """
        lines = []

        # --- 1. Per-case eval history ---
        case_evals = self.get_case_eval_history(case_id)
        if case_evals:
            lines.append(f"=== Eval History for case '{case_id}' ===")
            # Show most recent max_eval_records
            recent = case_evals[-max_eval_records:]
            for r in recent:
                defect_str = "; ".join(r.defects[:3]) if r.defects else "none"
                lines.append(
                    f"  [step={r.step} mode={r.mode}] "
                    f"reward={r.reward:.3f} decision={r.decision} "
                    f"action='{r.action_taken}' "
                    f"defects: {defect_str}"
                )
                if r.reasoning:
                    lines.append(f"    reasoning: {r.reasoning[:150]}")
        else:
            lines.append(f"No prior evaluation history for case '{case_id}'.")

        # --- 2. Global plan dialogue memory ---
        plan_summaries = self.get_plan_dialogue_summaries(top_k=max_plan_dialogues)
        if plan_summaries:
            lines.append("")
            lines.append("=== Top Successful Plan Dialogues (cross-case memory) ===")
            lines.extend(plan_summaries)
        else:
            lines.append("")
            lines.append("No successful plan dialogues in memory yet.")

        return "\n".join(lines)

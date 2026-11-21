"""
Search2Adapt MCTS Core — Node & Tree 数据结构

包含两套数据结构:
  1. MCTSNode / MCTSTree — 原始 action-level 搜索 (保留向后兼容)
  2. PlanNode / PlanTree — CoTS-style plan-level 协作搜索 (Option B)

--- Action-level (legacy) ---
  S: MaskState (image + current mask)
  A: Action (tool_id + params)
  P: Executor 执行动作后的状态转移
  R: Evaluator 给出的 LLM-based reward

--- Plan-level (CoTS Option B) ---
  Node = {RefinementPlan, Dialogue, plan_reward}
  Phase 1: Strategist(Planner) 提出完整计划, Evaluator(Critic) 评估计划质量
  Phase 2: Top-K 计划实际执行验证

--- Diversity UCT Extension ---
  UCT formula augmented with intrinsic diversity bonus:
    UCT(s_i) = V(s_i) + C * sqrt(ln(N_parent) / N_i) + λ / sqrt(count(tool_sig_i))

  where count(tool_sig_i) tracks how many nodes in the tree share the same
  ordered tool_id sequence (tool signature).  This encourages exploration of
  under-tried tool combinations beyond pure reward maximisation.

  Controlled by PlanTree.diversity_weight (λ, default 0.1).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from search2adapt.agents.base import Action, EvalResult, MaskState

# =============================================================================
#  1. MCTS Node
# =============================================================================


class MCTSNode:
    """
    搜索树节点

    每个节点存储:
      - state: 当前 MaskState
      - action: 到达此节点所执行的动作 (root 为 None)
      - parent: 父节点
      - children: 子节点列表
      - visits: 访问次数 N(s)
      - value: 累积价值 V(s) (所有经过此节点的 reward 的均值)
      - reward: 此节点自身的 evaluation reward
      - depth: 节点深度
    """

    def __init__(
        self,
        state: MaskState,
        action: Optional[Action] = None,
        parent: Optional["MCTSNode"] = None,
    ):
        self.state = state
        self.action = action  # 到达此节点的动作
        self.parent = parent
        self.children: List["MCTSNode"] = []

        # MCTS 统计量
        self.visits: int = 0  # N(s)
        self.total_value: float = 0.0  # 累积 reward 总和
        self.value: float = 0.0  # V(s) = total_value / visits
        self.reward: float = 0.0  # 此节点自身的 reward

        # 元数据
        self.depth: int = (parent.depth + 1) if parent else 0
        self.eval_result: Optional[EvalResult] = None  # Evaluator 的完整评估

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_fully_expanded(self) -> bool:
        """节点是否已被 Strategist 扩展过"""
        return len(self.children) > 0

    def uct_score(
        self,
        exploration_weight: float = 1.414,
        diversity_weight: float = 0.0,
        tool_usage_counts: Optional[Dict[Tuple, int]] = None,
    ) -> float:
        """
        UCT (Upper Confidence bounds for Trees) 公式 (with optional diversity bonus):
        UCT(s_i) = V(s_i) + w * sqrt(ln(N(parent)) / N(s_i))
                            + λ / sqrt(count(tool_sig_i))   [if diversity_weight > 0]

        Args:
            exploration_weight: 探索系数 w
            diversity_weight:   多样性奖励系数 λ (default 0 = disabled)
            tool_usage_counts:  {tool_signature_tuple: count} across the whole tree
        Returns:
            UCT 分数 (越高越优先选择)
        """
        if self.visits == 0:
            return float("inf")  # 未访问的节点具有最高优先级

        parent_visits = self.parent.visits if self.parent else self.visits
        if parent_visits == 0:
            # Parent hasn't been visited through backprop yet — treat
            # exploration term as 0 (pure exploitation)
            return self.value

        exploitation = self.value
        exploration = exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        diversity = 0.0
        if diversity_weight > 0.0 and tool_usage_counts:
            # Tool signature for MCTSNode: single-element tuple from action tool_id
            sig: Tuple = (self.action.tool_id,) if self.action else ()
            count = tool_usage_counts.get(sig, 1)
            diversity = diversity_weight / math.sqrt(max(count, 1))
        return exploitation + exploration + diversity

    def best_child(self, exploration_weight: float = 1.414) -> "MCTSNode":
        """选择 UCT 分数最高的子节点"""
        if not self.children:
            raise ValueError("No children to select from")
        return max(self.children, key=lambda c: c.uct_score(exploration_weight))

    def add_child(self, state: MaskState, action: Action) -> "MCTSNode":
        """添加子节点"""
        child = MCTSNode(state=state, action=action, parent=self)
        self.children.append(child)
        return child

    def update(self, reward: float):
        """
        更新节点统计量 (Backpropagation 时调用)
        V(s) = (V(s_prev) * (N-1) + r) / N
        """
        self.visits += 1
        self.total_value += reward
        self.value = self.total_value / self.visits

    def get_path(self) -> List["MCTSNode"]:
        """获取从 root 到当前节点的路径"""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_action_trajectory(self) -> List[Action]:
        """获取从 root 到当前节点的动作序列"""
        path = self.get_path()
        return [node.action for node in path if node.action is not None]

    def __repr__(self) -> str:
        action_str = self.action.tool_id if self.action else "ROOT"
        return (
            f"MCTSNode(action={action_str}, depth={self.depth}, "
            f"visits={self.visits}, value={self.value:.3f}, "
            f"children={len(self.children)})"
        )


# =============================================================================
#  2. MCTS Tree
# =============================================================================


class MCTSTree:
    """
    蒙特卡洛树搜索树
    管理搜索过程的整体状态
    """

    def __init__(
        self,
        root_state: MaskState,
        max_depth: int = 6,
        exploration_weight: float = 1.414,
        logger: Optional[logging.Logger] = None,
    ):
        self.root = MCTSNode(state=root_state)
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.logger = logger or logging.getLogger("MCTSTree")

        # 搜索统计
        self.total_iterations: int = 0
        self.best_reward: float = 0.0
        self.best_node: Optional[MCTSNode] = None

    # ---- Phase 1: Selection ----
    def select(self) -> MCTSNode:
        """
        从 root 沿着 UCT 最优路径选择到叶节点
        """
        node = self.root
        while not node.is_leaf:
            node = node.best_child(self.exploration_weight)
        return node

    # ---- Phase 4: Backpropagation ----
    def backpropagate(self, node: MCTSNode, reward: float):
        """
        将 reward 从叶节点反向传播到 root
        更新路径上所有节点的 visits 和 value
        """
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

        # 更新全局最优
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_node = node
            self.logger.info(
                f"New best path found! Reward={reward:.3f}, "
                f"Trajectory={[a.tool_id for a in node.get_action_trajectory()]}"
            )

        self.total_iterations += 1

    # ---- 结果提取 ----
    def get_best_trajectory(self, min_visits: int = 1) -> List[Action]:
        """
        获取最优动作序列 (基于最终 value 最高的叶节点)
        """
        if self.best_node is not None:
            return self.best_node.get_action_trajectory()

        # Fallback: 从 root 贪心选择 value 最高的路径
        node = self.root
        trajectory = []
        while not node.is_leaf:
            candidates = [c for c in node.children if c.visits >= min_visits]
            if not candidates:
                break
            best = max(candidates, key=lambda c: c.value)
            trajectory.append(best.action)
            node = best
        return trajectory

    def get_best_leaf(self, min_visits: int = 1) -> Optional[MCTSNode]:
        """获取 value 最高的叶节点"""
        if self.best_node is not None and self.best_node.visits >= min_visits:
            return self.best_node
        return None

    # ---- 树统计 ----
    def get_stats(self) -> Dict:
        """获取搜索树统计信息"""

        def _count_nodes(node: MCTSNode) -> int:
            return 1 + sum(_count_nodes(c) for c in node.children)

        def _max_depth(node: MCTSNode) -> int:
            if node.is_leaf:
                return node.depth
            return max(_max_depth(c) for c in node.children)

        total_nodes = _count_nodes(self.root)
        max_depth = _max_depth(self.root) if not self.root.is_leaf else 0

        return {
            "total_nodes": total_nodes,
            "total_iterations": self.total_iterations,
            "max_depth_reached": max_depth,
            "best_reward": self.best_reward,
            "root_visits": self.root.visits,
            "best_trajectory": (
                [a.to_dict() for a in self.best_node.get_action_trajectory()]
                if self.best_node
                else []
            ),
        }

    def print_tree(self, node: Optional[MCTSNode] = None, indent: int = 0):
        """打印搜索树 (调试用)"""
        if node is None:
            node = self.root
        prefix = "  " * indent
        self.logger.info(f"{prefix}{node}")
        for child in node.children:
            self.print_tree(child, indent + 1)


# =============================================================================
#  3. CoTS Plan-level Data Structures
# =============================================================================


@dataclass
class RefinementPlan:
    """
    完整的多步细化计划 (存储在 PlanNode 中)

    对应 CoTS 论文中 Alice 提出的 collaborative plan:
      Step 1: apply morphological closing on liver ...
      Step 2: run SAM3 re-segmentation on kidney slices ...
    """

    steps: List[Action]  # 有序工具链
    rationale: str = ""  # Strategist 的推理过程
    target_defects: List[str] = field(default_factory=list)  # 此计划针对的缺陷
    expected_outcome: str = ""  # 预期效果

    def to_dict(self) -> Dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "rationale": self.rationale,
            "target_defects": self.target_defects,
            "expected_outcome": self.expected_outcome,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RefinementPlan":
        return cls(
            steps=[Action.from_dict(s) for s in d.get("steps", [])],
            rationale=d.get("rationale", ""),
            target_defects=d.get("target_defects", []),
            expected_outcome=d.get("expected_outcome", ""),
        )


@dataclass
class Dialogue:
    """
    Strategist-Evaluator 对话记录 (存储在 PlanNode 中)

    对应 CoTS 论文中每个节点的 Alice-Bob 对话:
      Alice Output: "I generate a plan, now..."
      Bob Output: "But I think we ..." + Reward: 0.6
    """

    strategist_proposal: str = ""  # Strategist(Planner) 的计划提案描述
    evaluator_critique: str = ""  # Evaluator(Critic) 的评估与建议
    evaluator_reward: float = 0.0  # 计划级 reward [0, 1]
    evaluator_suggestions: List[str] = field(
        default_factory=list
    )  # Evaluator 提出的改进建议 (供子节点扩展时参考)

    def to_dict(self) -> Dict:
        return {
            "strategist_proposal": self.strategist_proposal,
            "evaluator_critique": self.evaluator_critique,
            "evaluator_reward": self.evaluator_reward,
            "evaluator_suggestions": self.evaluator_suggestions,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Dialogue":
        return cls(
            strategist_proposal=d.get("strategist_proposal", ""),
            evaluator_critique=d.get("evaluator_critique", ""),
            evaluator_reward=d.get("evaluator_reward", 0.0),
            evaluator_suggestions=d.get("evaluator_suggestions", []),
        )


# =============================================================================
#  4. CoTS Plan Node
# =============================================================================


class PlanNode:
    """
    CoTS-style 计划搜索树节点

    与 MCTSNode 的区别:
      - MCTSNode: 存储单个 Action + 执行后的 MaskState
      - PlanNode: 存储完整 RefinementPlan + Strategist-Evaluator Dialogue
      - PlanNode 在 Phase 1 不执行任何工具, 仅通过 LLM 评估计划质量
      - Top-K 节点在 Phase 2 才实际执行

    每个节点存储:
      - plan: 完整细化计划 (root 为空计划)
      - dialogue: Strategist-Evaluator 对话
      - parent/children: 树结构
      - MCTS 统计量: visits, value, plan_reward
      - 执行验证结果 (Phase 2, 仅 top-K 节点)
    """

    def __init__(
        self,
        plan: Optional[RefinementPlan] = None,
        dialogue: Optional[Dialogue] = None,
        state: Optional[MaskState] = None,
        parent: Optional["PlanNode"] = None,
    ):
        self.plan = plan or RefinementPlan(steps=[])
        self.dialogue = dialogue or Dialogue()
        self.state = state  # 关联的 MaskState (root 必须有, 子节点可选)
        self.parent = parent
        self.children: List["PlanNode"] = []

        # MCTS 统计量
        self.visits: int = 0  # N(s)
        self.total_value: float = 0.0  # 累积 plan_reward 总和
        self.value: float = 0.0  # V(s) = total_value / visits
        self.plan_reward: float = 0.0  # 此节点自身的计划级 reward

        # 元数据
        self.depth: int = (parent.depth + 1) if parent else 0

        # Phase 2 执行验证结果 (仅 top-K 节点)
        self.executed: bool = False
        self.execution_reward: Optional[float] = None  # 实际 mask 质量 reward
        self.execution_state: Optional[MaskState] = None  # 执行后的 MaskState
        self.execution_eval: Optional[EvalResult] = None  # 完整评估结果

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_fully_expanded(self) -> bool:
        """节点是否已被 Strategist 扩展过"""
        return len(self.children) > 0

    def uct_score(
        self,
        exploration_weight: float = 1.414,
        diversity_weight: float = 0.0,
        tool_usage_counts: Optional[Dict[Tuple, int]] = None,
    ) -> float:
        """
        UCT (Upper Confidence bounds for Trees) 公式 (with optional diversity bonus):
        UCT(s_i) = V(s_i) + w * sqrt(ln(N(parent)) / N(s_i))
                            + λ / sqrt(count(tool_sig_i))   [if diversity_weight > 0]

        Args:
            exploration_weight: 探索系数 w
            diversity_weight:   多样性奖励系数 λ (default 0 = disabled)
            tool_usage_counts:  {tool_signature_tuple: count} across the whole tree
        """
        if self.visits == 0:
            return float("inf")

        parent_visits = self.parent.visits if self.parent else self.visits
        if parent_visits == 0:
            return self.value

        exploitation = self.value
        exploration = exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        diversity = 0.0
        if diversity_weight > 0.0 and tool_usage_counts:
            sig: Tuple = tuple(s.tool_id for s in self.plan.steps)
            count = tool_usage_counts.get(sig, 1)
            diversity = diversity_weight / math.sqrt(max(count, 1))
        return exploitation + exploration + diversity

    def best_child(
        self,
        exploration_weight: float = 1.414,
        diversity_weight: float = 0.0,
        tool_usage_counts: Optional[Dict[Tuple, int]] = None,
    ) -> "PlanNode":
        """选择 UCT 分数最高的子节点"""
        if not self.children:
            raise ValueError("No children to select from")
        return max(
            self.children,
            key=lambda c: c.uct_score(
                exploration_weight,
                diversity_weight=diversity_weight,
                tool_usage_counts=tool_usage_counts,
            ),
        )

    def add_child(
        self,
        plan: RefinementPlan,
        dialogue: Dialogue,
        state: Optional[MaskState] = None,
    ) -> "PlanNode":
        """添加子节点"""
        child = PlanNode(plan=plan, dialogue=dialogue, state=state, parent=self)
        self.children.append(child)
        return child

    def update(self, reward: float):
        """
        更新节点统计量 (Backpropagation 时调用)
        V(s) = (V(s_prev) * (N-1) + r) / N
        """
        self.visits += 1
        self.total_value += reward
        self.value = self.total_value / self.visits

    def get_path(self) -> List["PlanNode"]:
        """获取从 root 到当前节点的路径"""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_plan_trajectory(self) -> List[RefinementPlan]:
        """获取从 root 到当前节点的计划序列"""
        path = self.get_path()
        return [node.plan for node in path if node.plan.steps]

    def get_final_plan(self) -> RefinementPlan:
        """
        获取最终计划 — 叶节点的计划就是最终完整计划
        (每轮子节点的计划是对父节点计划的细化/修正, 最终计划在叶节点)
        """
        return self.plan

    def get_full_action_sequence(self) -> List[Action]:
        """获取最终计划的完整工具链"""
        return list(self.plan.steps)

    def __repr__(self) -> str:
        plan_str = (
            f"[{','.join(s.tool_id for s in self.plan.steps)}]"
            if self.plan.steps
            else "ROOT"
        )
        exec_str = (
            f", exec_reward={self.execution_reward:.3f}"
            if self.execution_reward is not None
            else ""
        )
        return (
            f"PlanNode(plan={plan_str}, depth={self.depth}, "
            f"visits={self.visits}, value={self.value:.3f}, "
            f"plan_reward={self.plan_reward:.3f}{exec_str}, "
            f"children={len(self.children)})"
        )


# =============================================================================
#  5. CoTS Plan Tree
# =============================================================================


class PlanTree:
    """
    CoTS-style 计划搜索树

    与 MCTSTree 的区别:
      - MCTSTree: 节点是单个 Action, 搜索探索动作空间
      - PlanTree: 节点是完整 RefinementPlan, 搜索探索计划空间
      - Phase 1: 纯 LLM 评估, 无实际执行
      - Phase 2: Top-K 计划实际执行验证
    """

    def __init__(
        self,
        root_state: MaskState,
        max_depth: int = 4,
        exploration_weight: float = 1.414,
        diversity_weight: float = 0.1,
        logger: Optional[logging.Logger] = None,
    ):
        self.root = PlanNode(state=root_state)
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.diversity_weight = diversity_weight
        self.logger = logger or logging.getLogger("PlanTree")

        # 搜索统计
        self.total_iterations: int = 0
        self.best_plan_reward: float = 0.0
        self.best_plan_node: Optional[PlanNode] = None

        # Phase 2 执行验证统计
        self.best_execution_reward: float = 0.0
        self.best_execution_node: Optional[PlanNode] = None

        # Diversity tracking: {tool_signature_tuple: count_across_tree}
        self._tool_usage_counts: Dict[Tuple, int] = {}

    # ---- Phase 1: Selection ----
    def select(self) -> PlanNode:
        """
        从 root 沿着 UCT 最优路径选择到叶节点
        Passes current tool_usage_counts for diversity-aware UCT.
        """
        node = self.root
        while not node.is_leaf:
            node = node.best_child(
                self.exploration_weight,
                diversity_weight=self.diversity_weight,
                tool_usage_counts=self._tool_usage_counts,
            )
        return node

    # ---- Phase 1: Backpropagation (plan-level) ----
    def backpropagate(self, node: PlanNode, reward: float):
        """
        将 plan_reward 从叶节点反向传播到 root
        更新路径上所有节点的 visits 和 value.
        Also updates _tool_usage_counts for diversity tracking.
        """
        # Update diversity counter for this node's tool signature
        sig: Tuple = tuple(s.tool_id for s in node.plan.steps)
        if sig:  # Skip root (empty plan)
            self._tool_usage_counts[sig] = self._tool_usage_counts.get(sig, 0) + 1

        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

        # 更新全局最优
        if reward > self.best_plan_reward:
            self.best_plan_reward = reward
            self.best_plan_node = node
            self.logger.info(
                f"New best plan found! Plan reward={reward:.3f}, "
                f"Steps={[s.tool_id for s in node.plan.steps]}"
            )

        self.total_iterations += 1

    # ---- Top-K Extraction for Phase 2 ----
    def get_top_k_plans(self, k: int = 3, min_visits: int = 1) -> List[PlanNode]:
        """
        提取搜索树中 value 最高的 k 个叶节点 (用于 Phase 2 执行验证)

        策略: 遍历所有叶节点, 按 value 降序排列, 取前 k 个
        """
        leaves = []
        self._collect_leaves(self.root, leaves)

        # 过滤 min_visits
        qualified = [n for n in leaves if n.visits >= min_visits and n.plan.steps]
        # 按 value 降序排列
        qualified.sort(key=lambda n: n.value, reverse=True)
        return qualified[:k]

    def _collect_leaves(self, node: PlanNode, result: List[PlanNode]):
        """递归收集所有叶节点"""
        if node.is_leaf:
            result.append(node)
        else:
            for child in node.children:
                self._collect_leaves(child, result)

    # ---- 记录 Phase 2 执行结果 ----
    def record_execution(
        self, node: PlanNode, reward: float, state: MaskState, eval_result: EvalResult
    ):
        """记录某个计划节点的执行验证结果"""
        node.executed = True
        node.execution_reward = reward
        node.execution_state = state
        node.execution_eval = eval_result

        if reward > self.best_execution_reward:
            self.best_execution_reward = reward
            self.best_execution_node = node
            self.logger.info(
                f"New best execution! Reward={reward:.3f}, "
                f"Steps={[s.tool_id for s in node.plan.steps]}"
            )

    # ---- 结果提取 ----
    def get_best_plan_trajectory(self, min_visits: int = 1) -> List[Action]:
        """
        获取最优动作序列:
          1. 优先返回 Phase 2 执行验证最优的计划
          2. 否则返回 Phase 1 plan_reward 最优的计划
        """
        # Phase 2 结果优先
        if self.best_execution_node is not None:
            return self.best_execution_node.get_full_action_sequence()

        # Fallback: Phase 1 最优计划
        if self.best_plan_node is not None:
            return self.best_plan_node.get_full_action_sequence()

        # Fallback: 贪心路径
        node = self.root
        while not node.is_leaf:
            candidates = [c for c in node.children if c.visits >= min_visits]
            if not candidates:
                break
            node = max(candidates, key=lambda c: c.value)
        return node.get_full_action_sequence()

    def get_best_node(self) -> Optional[PlanNode]:
        """获取最优节点 (Phase 2 优先)"""
        if self.best_execution_node is not None:
            return self.best_execution_node
        return self.best_plan_node

    def get_best_state(self) -> Optional[MaskState]:
        """获取最优 MaskState (Phase 2 执行后的状态)"""
        if self.best_execution_node is not None:
            return self.best_execution_node.execution_state
        return self.root.state

    def get_best_reward(self) -> float:
        """获取最优 reward (Phase 2 优先)"""
        if self.best_execution_node is not None:
            return self.best_execution_reward
        return self.best_plan_reward

    # ---- 树统计 ----
    def get_stats(self) -> Dict:
        """获取搜索树统计信息"""

        def _count_nodes(node: PlanNode) -> int:
            return 1 + sum(_count_nodes(c) for c in node.children)

        def _max_depth(node: PlanNode) -> int:
            if node.is_leaf:
                return node.depth
            return max(_max_depth(c) for c in node.children)

        total_nodes = _count_nodes(self.root)
        max_depth = _max_depth(self.root) if not self.root.is_leaf else 0

        # 收集已执行的节点
        executed_nodes = []
        self._collect_executed(self.root, executed_nodes)

        return {
            "total_nodes": total_nodes,
            "total_iterations": self.total_iterations,
            "max_depth_reached": max_depth,
            "best_plan_reward": self.best_plan_reward,
            "best_execution_reward": self.best_execution_reward,
            "executed_plans": len(executed_nodes),
            "root_visits": self.root.visits,
            "diversity_weight": self.diversity_weight,
            "unique_tool_signatures": len(self._tool_usage_counts),
            "best_plan": (
                [s.to_dict() for s in self.best_plan_node.plan.steps]
                if self.best_plan_node
                else []
            ),
            "best_executed_plan": (
                [s.to_dict() for s in self.best_execution_node.plan.steps]
                if self.best_execution_node
                else []
            ),
        }

    def _collect_executed(self, node: PlanNode, result: List[PlanNode]):
        """递归收集所有已执行的节点"""
        if node.executed:
            result.append(node)
        for child in node.children:
            self._collect_executed(child, result)

    def print_tree(self, node: Optional[PlanNode] = None, indent: int = 0):
        """打印搜索树 (调试用)"""
        if node is None:
            node = self.root
        prefix = "  " * indent
        self.logger.info(f"{prefix}{node}")
        for child in node.children:
            self.print_tree(child, indent + 1)

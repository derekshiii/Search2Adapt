"""
Search2Adapt Agent Base Classes
定义四个 Agent 角色的基类和通信协议

Multi-Agent 架构 (受临床 MDT 启发):
  - Perceptor (Scout): 领域态势感知
  - Strategist (Planner): MCTS 策略搜索 + 动作选择
  - Executor (Actor): 工具执行
  - Evaluator (Critic): LLM 驱动的 Reward 函数
"""

import base64
import copy
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np
from openai import OpenAI

from search2adapt.config import LLMConfig, Search2AdaptConfig

# =============================================================================
#  1. 状态定义 (State for MDP)
# =============================================================================


@dataclass
class MaskState:
    """
    MCTS 中的状态 S = (image, mask, metadata)
    对应论文: 状态空间 S = 当前图像 x 及其对应的中间态伪标签 m_t
    """

    image_path: str  # 原始图像路径 (可能经过 CLAHE 预处理)
    mask_path: str  # 当前中间态伪标签路径
    case_id: str  # 样本 ID
    step: int = 0  # 当前步数 t

    # 缓存的几何特征 (供 Evaluator 使用, 由代码提取后文本化给 LLM)
    cached_features: Optional[Dict] = None

    def clone(self, new_mask_path: str, new_step: int) -> "MaskState":
        """创建状态副本, 更新 mask 和步数"""
        return MaskState(
            image_path=self.image_path,
            mask_path=new_mask_path,
            case_id=self.case_id,
            step=new_step,
            cached_features=copy.deepcopy(self.cached_features)
            if self.cached_features
            else None,
        )


@dataclass
class Action:
    """
    MCTS 中的动作 A = (tool_id, params)
    对应论文: 动作空间 A = Toolbox 中的工具集合
    """

    tool_id: str  # 工具 ID, 对应 TOOL_REGISTRY 的 key
    params: Dict = field(default_factory=dict)
    description: str = ""  # LLM 生成的动作描述 (用于日志)

    def to_dict(self) -> Dict:
        return {
            "tool_id": self.tool_id,
            "params": self.params,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Action":
        return cls(
            tool_id=d["tool_id"],
            params=d.get("params", {}),
            description=d.get("description", ""),
        )


@dataclass
class EvalResult:
    """Evaluator 返回的评估结果"""

    reward: float  # 归一化 reward [0, 1]
    scores: Dict[str, float] = field(default_factory=dict)  # 各维度得分
    reasoning: str = ""  # LLM 给出的推理过程
    decision: str = "CONTINUE"  # ACCEPT / REJECT / CONTINUE
    suggested_actions: List[str] = field(default_factory=list)  # 建议的后续动作


# =============================================================================
#  2. LLM 调用基础设施
# =============================================================================


class LLMCaller:
    """LLM API 调用封装"""

    def __init__(self, config: LLMConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("LLMCaller")
        # Unset SOCKS proxy (all_proxy) to avoid httpx crash — SiliconFlow works via HTTP proxy
        from urllib.parse import urlparse

        import httpx

        env_proxies = os.environ.copy()
        env_proxies.pop("all_proxy", None)
        env_proxies.pop("ALL_PROXY", None)

        # A locally served model must never be sent through an HTTP proxy: when
        # http_proxy is exported the proxy would swallow the loopback request.
        host = (urlparse(config.base_url).hostname or "").lower()
        is_loopback = host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

        proxy = None
        if not is_loopback:
            proxy = (
                env_proxies.get("http_proxy")
                or env_proxies.get("HTTP_PROXY")
                or env_proxies.get("https_proxy")
                or env_proxies.get("HTTPS_PROXY")
            )

        http_client = httpx.Client(
            proxy=proxy,
            # trust_env=False for loopback, otherwise httpx re-reads http_proxy
            # from the environment and routes the local request through it anyway.
            trust_env=not is_loopback,
            timeout=httpx.Timeout(config.timeout, connect=30.0),
        )
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            http_client=http_client,
        )

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        no_think: bool = False,
    ) -> str:
        """
        调用 LLM/VLM
        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            images: 图像路径列表 (如果是 VLM 多模态调用)
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认 max_tokens
            no_think: 如果为 True, 禁用 Qwen3 的 think 模式 (chat_template_kwargs)
        Returns:
            LLM 返回的文本
        """
        messages = [{"role": "system", "content": system_prompt}]

        # 构建 user message
        if images:
            content = []
            for img_path in images:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(img_path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                )
            content.append({"type": "text", "text": user_prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        max_retries = 3
        backoff_base = 2.0  # seconds

        # Build extra kwargs for no-think mode (SiliconFlow / Qwen3)
        extra_kwargs: Dict[str, Any] = {}
        if no_think:
            extra_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature or self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    **extra_kwargs,
                )
                text = response.choices[0].message.content or ""
                text = text.strip()

                # Strip Qwen3 <think>...</think> reasoning tags
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

                return text
            except Exception as e:
                self.logger.warning(
                    f"LLM call attempt {attempt}/{max_retries} failed: {e}"
                )
                if attempt == max_retries:
                    self.logger.error(
                        f"LLM call failed after {max_retries} attempts: {e}"
                    )
                    raise
                wait = backoff_base**attempt
                self.logger.info(f"Retrying in {wait:.1f}s...")
                time.sleep(wait)

        # Should never reach here, but satisfy type checker
        raise RuntimeError("LLM call failed: exhausted all retries")


# =============================================================================
#  3. Mask 几何特征提取 (供 Evaluator 文本化使用)使用VLM直接理解就可以
# =============================================================================


def extract_mask_features(
    mask_path: str, organ_label_map: Optional[Dict[str, int]] = None
) -> Dict:
    """
    从 mask 中提取几何/拓扑特征, 供 LLM Evaluator 作为文本输入
    LLM 看不懂纯图像数组, 需要代码先提取再文本化

    Returns:
        features dict 包含:
        - per_label: 每个 label 的体素数, 连通域数, 占空比, bbox 范围等
        - global: 总非零体素占比, label 数量等
    """
    obj = nib.load(mask_path)
    data = np.round(obj.get_fdata()).astype(np.int16)
    shape = data.shape
    total_voxels = int(np.prod(shape))

    from scipy.ndimage import label as nd_label

    unique_labels = [int(l) for l in np.unique(data) if l != 0]
    per_label = {}

    # 构建 label_id -> organ_name 反向映射
    id_to_name = {}
    if organ_label_map:
        id_to_name = {v: k for k, v in organ_label_map.items()}

    for lid in unique_labels:
        binary = data == lid
        voxel_count = int(binary.sum())
        occupancy = voxel_count / total_voxels

        # 连通域
        labeled, num_cc = nd_label(binary)

        # BBox
        coords = np.where(binary)
        bbox_min = [int(c.min()) for c in coords]
        bbox_max = [int(c.max()) for c in coords]

        name = id_to_name.get(lid, f"label_{lid}")
        per_label[name] = {
            "label_id": lid,
            "voxel_count": voxel_count,
            "occupancy_ratio": round(occupancy, 6),
            "connected_components": num_cc,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_size": [mx - mn for mn, mx in zip(bbox_min, bbox_max)],
        }

    features = {
        "volume_shape": list(shape),
        "total_voxels": total_voxels,
        "num_labels": len(unique_labels),
        "total_foreground_ratio": round(float((data > 0).sum()) / total_voxels, 6),
        "per_label": per_label,
    }
    return features


def features_to_text(features: Dict) -> str:
    """将提取的特征转为 LLM 可读文本"""
    lines = [
        f"Volume shape: {features['volume_shape']}",
        f"Number of labels: {features['num_labels']}",
        f"Total foreground ratio: {features['total_foreground_ratio']:.4f}",
        "",
        "Per-label details:",
    ]
    for name, info in features["per_label"].items():
        lines.append(
            f"  [{name}] (ID={info['label_id']}): "
            f"voxels={info['voxel_count']}, "
            f"occupancy={info['occupancy_ratio']:.4f}, "
            f"connected_components={info['connected_components']}, "
            f"bbox_size={info['bbox_size']}"
        )
    return "\n".join(lines)


# =============================================================================
#  4. Agent 基类
# =============================================================================


# =============================================================================
#  4.5. Axial Axis Inference (robust to isotropic volumes)
# =============================================================================


def infer_axial_axis(volume_shape: tuple) -> int:
    """
    Infer the axial (slice) axis from a 3D medical image shape.

    Heuristic: the axial axis typically has the fewest slices.
    For ambiguous cases (e.g., isotropic 256x256x256), default to axis 2
    (NIfTI RAS+ convention where axis 2 is Superior-Inferior).

    Args:
        volume_shape: 3-tuple of spatial dimensions

    Returns:
        integer axis index (0, 1, or 2)
    """
    if len(volume_shape) != 3:
        return 2  # Fallback for non-3D data

    min_dim = min(volume_shape)
    min_axes = [i for i in range(3) if volume_shape[i] == min_dim]

    if len(min_axes) == 1:
        return min_axes[0]

    # Ambiguous (e.g., isotropic): default to axis 2 (NIfTI convention)
    if 2 in min_axes:
        return 2
    return min_axes[0]


# =============================================================================
#  5. Agent 基类
# =============================================================================


class BaseAgent(ABC):
    """所有 Agent 的抽象基类"""

    def __init__(
        self,
        name: str,
        config: Search2AdaptConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.name = name
        self.config = config
        self.logger = logger or logging.getLogger(name)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Agent 的核心执行逻辑"""
        raise NotImplementedError

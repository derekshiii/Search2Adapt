"""
Search2Adapt Configuration Management

Centralised management of paths, API endpoints, hyperparameters and environment
variables. Every machine-specific value is read from environment variables so
that the repository can be published without leaking internal paths or keys.

Environment variables
---------------------
LLM / VLM endpoint
    SEARCH2ADAPT_LLM_API_KEY   (falls back to OPENAI_API_KEY)
    SEARCH2ADAPT_LLM_BASE_URL  (falls back to OPENAI_BASE_URL, then http://localhost:8000/v1)
    SEARCH2ADAPT_LLM_MODEL     (default: qwen3.5-0.8b)
    SEARCH2ADAPT_VLM_API_KEY / _BASE_URL / _MODEL  (fall back to the LLM values)

Vision foundation models
    SEARCH2ADAPT_BIOMEDPARSE_ROOT       checkout of the BiomedParse repository
    SEARCH2ADAPT_BIOMEDPARSE_CHECKPOINT  3D BiomedParse checkpoint
    SEARCH2ADAPT_BIOMEDPARSE_CONFIG_DIR  hydra config dir (default: <root>/configs/model)
    SEARCH2ADAPT_BIOMEDPARSE_PYTHON     interpreter of the BiomedParse environment
    SEARCH2ADAPT_SAM3_ROOT              checkout of the SAM3 repository
    SEARCH2ADAPT_SAM3_CHECKPOINT        SAM3 checkpoint
    SEARCH2ADAPT_SAM3_PYTHON            interpreter of the SAM3 environment

nnU-Net (knowledge distillation phase)
    nnUNet_raw / nnUNet_preprocessed / nnUNet_results   (standard nnU-Net variables)
    SEARCH2ADAPT_NNUNET_BIN             bin directory of the nnU-Net environment
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# =============================================================================
#  1. Path Configuration
# =============================================================================

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_VFM_DIR = os.path.join(_PACKAGE_DIR, "vfm")


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _env_path(name: str, default: str = "") -> str:
    return os.path.abspath(os.path.expanduser(_env(name, default)))


@dataclass
class PathConfig:
    """Locations of every external dependency used by the pipeline."""

    # --- BiomedParse (3D) ---
    biomedparse_root: str = ""
    biomedparse_checkpoint: str = ""
    biomedparse_config_dir: str = ""
    biomedparse_python: str = ""
    biomedparse_script: str = os.path.join(_VFM_DIR, "biomedparse_3d_standalone.py")

    # --- SAM3 (3D) ---
    sam3_root: str = ""
    sam3_checkpoint: str = ""
    sam3_python: str = ""
    sam3_script: str = os.path.join(_VFM_DIR, "sam3_3d_standalone.py")

    # --- nnU-Net ---
    nnunet_raw: str = ""
    nnunet_preprocessed: str = ""
    nnunet_results: str = ""
    nnunet_env_path: Optional[str] = None

    def __post_init__(self):
        self.biomedparse_root = _env_path(
            "SEARCH2ADAPT_BIOMEDPARSE_ROOT", self.biomedparse_root
        )
        self.biomedparse_checkpoint = _env_path(
            "SEARCH2ADAPT_BIOMEDPARSE_CHECKPOINT", self.biomedparse_checkpoint
        )
        self.biomedparse_config_dir = _env_path(
            "SEARCH2ADAPT_BIOMEDPARSE_CONFIG_DIR",
            self.biomedparse_config_dir
            or (
                os.path.join(self.biomedparse_root, "configs", "model")
                if self.biomedparse_root
                else ""
            ),
        )
        self.biomedparse_python = _env_path(
            "SEARCH2ADAPT_BIOMEDPARSE_PYTHON",
            self.biomedparse_python or sys.executable,
        )
        self.biomedparse_script = _env_path(
            "SEARCH2ADAPT_BIOMEDPARSE_SCRIPT", self.biomedparse_script
        )

        self.sam3_root = _env_path("SEARCH2ADAPT_SAM3_ROOT", self.sam3_root)
        self.sam3_checkpoint = _env_path(
            "SEARCH2ADAPT_SAM3_CHECKPOINT", self.sam3_checkpoint
        )
        self.sam3_python = _env_path(
            "SEARCH2ADAPT_SAM3_PYTHON", self.sam3_python or sys.executable
        )
        self.sam3_script = _env_path("SEARCH2ADAPT_SAM3_SCRIPT", self.sam3_script)

        self.nnunet_raw = _env_path("nnUNet_raw", self.nnunet_raw)
        self.nnunet_preprocessed = _env_path(
            "nnUNet_preprocessed", self.nnunet_preprocessed
        )
        self.nnunet_results = _env_path("nnUNet_results", self.nnunet_results)
        self.nnunet_env_path = _env_path(
            "SEARCH2ADAPT_NNUNET_BIN", self.nnunet_env_path or ""
        ) or None

    def validate(self, logger: Optional[logging.Logger] = None) -> bool:
        """Warn about missing optional paths; returns True when nothing is broken."""
        checks = {
            "BiomedParse checkpoint": self.biomedparse_checkpoint,
            "BiomedParse root": self.biomedparse_root,
            "BiomedParse interpreter": self.biomedparse_python,
            "SAM3 checkpoint": self.sam3_checkpoint,
            "SAM3 root": self.sam3_root,
            "SAM3 interpreter": self.sam3_python,
        }
        if self.nnunet_env_path:
            checks["nnU-Net environment"] = self.nnunet_env_path

        all_ok = True
        for name, path in checks.items():
            if path and not os.path.exists(path):
                msg = f"[PathConfig] WARNING: {name} not found at {path}"
                logger.warning(msg) if logger else print(msg)
                all_ok = False

        missing_nnunet = [
            key
            for key, value in {
                "nnUNet_raw": self.nnunet_raw,
                "nnUNet_preprocessed": self.nnunet_preprocessed,
                "nnUNet_results": self.nnunet_results,
            }.items()
            if not value
        ]
        if missing_nnunet:
            msg = (
                "[PathConfig] WARNING: nnU-Net environment variables are not set "
                f"({', '.join(missing_nnunet)}); the distillation phase is disabled"
            )
            logger.warning(msg) if logger else print(msg)

        return all_ok


# =============================================================================
#  2. LLM / VLM API Configuration
# =============================================================================


@dataclass
class LLMConfig:
    """OpenAI-compatible endpoint used by Strategist / Evaluator."""

    api_key: str = ""
    base_url: str = "http://localhost:8000/v1"
    model: str = "qwen3.5-0.8b"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120  # seconds

    def __post_init__(self):
        self.api_key = _env("SEARCH2ADAPT_LLM_API_KEY") or _env("OPENAI_API_KEY")
        self.base_url = (
            _env("SEARCH2ADAPT_LLM_BASE_URL")
            or _env("OPENAI_BASE_URL")
            or self.base_url
        )
        self.model = _env("SEARCH2ADAPT_LLM_MODEL", self.model)


@dataclass
class VLMConfig:
    """OpenAI-compatible endpoint used for visual evaluation."""

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120

    def __post_init__(self):
        self.api_key = _env("SEARCH2ADAPT_VLM_API_KEY")
        self.base_url = _env("SEARCH2ADAPT_VLM_BASE_URL")
        self.model = _env("SEARCH2ADAPT_VLM_MODEL")


# =============================================================================
#  3. MCTS Hyperparameters
# =============================================================================


@dataclass
class MCTSConfig:
    """Monte-Carlo tree search hyper-parameters.

    The defaults follow the paper: the CAVF exploration weight is sqrt(2), the
    diversity coefficient lambda is 0.1, four candidate plans are proposed and
    the top-4 plans are executed for verification.
    """

    # --- CAVF ---
    exploration_weight: float = 2**0.5  # w, the classic sqrt(2) UCT coefficient
    diversity_weight: float = 0.1  # lambda, bonus = lambda / sqrt(count(tool_sig))

    # --- tree structure ---
    max_depth: int = 6  # maximum tool-chain length
    branching_factor: int = 4  # candidate plan variants per expansion
    max_iterations: int = 50  # search iterations per case

    # --- convergence ---
    reward_threshold: float = 0.85  # early stop once the reward is reached
    min_visits_for_best: int = 2  # minimum visit count for the best path

    # --- CoTS plan-level search ---
    top_k_execute: int = 4  # top-K plans executed for verification
    plan_reward_threshold: float = 0.80  # early stop of the phase-1 plan search
    execution_threshold: float = 0.85  # acceptance threshold of phase 2
    viability_check: bool = True  # monitor plan viability during execution
    viability_threshold: float = 0.4  # abort a plan below this viability
    plan_max_depth: int = 4  # depth of the plan search tree
    plan_max_iterations: int = 20  # iterations of the plan search

    # --- CoT reflection ---
    enable_cot_reflection: bool = True

    # --- ablation switches ---
    enable_mcts_search: bool = True  # False = greedy agent (no tree search)
    enable_perceptor_context: bool = True
    enable_refinement: bool = True  # False = raw VFM pseudo-label (no adaptation)


# =============================================================================
#  4. Toolbox Configuration
# =============================================================================


@dataclass
class ToolboxConfig:
    """Toolbox defaults."""

    toolset_mode: str = "full"

    def get_allowed_categories(self) -> set:
        if self.toolset_mode == "vfm_only":
            return {"vfm_proposer"}
        if self.toolset_mode == "vfm_postprocess":
            return {"vfm_proposer", "postprocessing"}
        return {"vfm_proposer", "postprocessing", "preprocessing", "diagnostic"}

    # --- BiomedParse ---
    bp_slice_batch_size: int = 8
    bp_interpolate_size: int = 512
    bp_nms_threshold: float = 0.5
    bp_score_threshold: float = 0.5

    # --- SAM3 ---
    sam3_gpus: List[int] = field(default_factory=lambda: [0, 1])

    # --- SymmetryChecker ---
    sym_vol_threshold: float = 0.2
    sym_dice_threshold: float = 0.1

    # --- ZoomROI ---
    zoom_margin: int = 10

    # --- morphological operations (MONAI dilate/erode) ---
    morph_filter_size: int = 3
    morph_iterations: int = 1


# =============================================================================
#  5. Memory / Perceptor Configuration
# =============================================================================


@dataclass
class PerceptorConfig:
    """Perceptor agent hyper-parameters."""

    grid_size: int = 3  # N x N grid sent to the VLM
    num_cases: int = 5  # randomly sampled cases
    slices_per_case: int = 9  # slices sampled per case
    cleanup_temp: bool = False  # delete the temporary grid images


@dataclass
class MemoryConfig:
    """Dynamic strategy prior memory."""

    memory_file: str = "strategy_memory.json"  # relative to result_dir
    max_trajectories: int = 500
    prior_top_k: int = 5  # number of frequent successful paths kept as priors
    prior_min_reward: float = 0.7  # minimum reward to enter the memory
    prior_accept_threshold: float = 0.75  # minimum reward to adopt a prior directly


# =============================================================================
#  6. Knowledge Distillation Configuration
# =============================================================================


@dataclass
class DistillationConfig:
    """nnU-Net knowledge distillation settings."""

    planner: str = "ResEncUNetPlanner"
    plans_identifier: str = "nnUNetResEncUNetPlans"
    fold: str = "all"
    num_epochs: int = 300
    pretrained_weights: Optional[str] = None  # source-domain checkpoint


# =============================================================================
#  7. Global Configuration
# =============================================================================


@dataclass
class Search2AdaptConfig:
    """Global configuration aggregating every sub-configuration."""

    experiment_name: str = "search2adapt_run"
    result_dir: str = "./S2A_Results"
    gpus: List[int] = field(default_factory=lambda: [0, 1])
    target_organs: str = ""
    target_data_path: str = ""

    paths: PathConfig = field(default_factory=PathConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    toolbox: ToolboxConfig = field(default_factory=ToolboxConfig)
    perceptor: PerceptorConfig = field(default_factory=PerceptorConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    max_workers: int = 8
    log_level: str = "INFO"
    debug: bool = False

    def __post_init__(self):
        if not self.vlm.api_key and self.llm.api_key:
            self.vlm.api_key = self.llm.api_key
        if not self.vlm.base_url and self.llm.base_url:
            self.vlm.base_url = self.llm.base_url
        if not self.vlm.model and self.llm.model:
            self.vlm.model = self.llm.model

    def validate(self, logger: Optional[logging.Logger] = None) -> bool:
        ok = True

        if not self.target_organs:
            msg = "[Config] ERROR: target_organs is empty"
            logger.error(msg) if logger else print(msg)
            ok = False

        if not self.target_data_path:
            msg = "[Config] ERROR: target_data_path is empty"
            logger.error(msg) if logger else print(msg)
            ok = False

        if not self.llm.api_key:
            msg = (
                "[Config] ERROR: no LLM API key. Set SEARCH2ADAPT_LLM_API_KEY "
                "(or OPENAI_API_KEY) before running the pipeline."
            )
            logger.error(msg) if logger else print(msg)
            ok = False

        ok = self.paths.validate(logger) and ok
        return ok

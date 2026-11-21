"""
Search2Adapt Ablation Experiment Configuration Presets

The preset names mirror the ablation rows of Table 5 in the paper:

    Search2Adapt w/o Perceptor
    Search2Adapt w/o CoTS
    Search2Adapt w/o CoT Diversity
    Search2Adapt w/o CoT Reflection
    Search2Adapt w/o VFM Postprocess
    Search2Adapt

Table 5's "No Adaptation" row is deliberately absent: the paper defines it as
the source model evaluated directly on the target domain, which this pipeline
never produces — there is no source data or source inference step in SFUDA.
Reproduce that row by running the pretrained source nnU-Net over the target
images and scoring the output with ``search2adapt evaluate``.

Usage:
    from search2adapt.ablation_configs import apply_ablation_config

    config = Search2AdaptConfig(...)
    config = apply_ablation_config(config, "wo_cot_diversity")
"""

from search2adapt.config import Search2AdaptConfig

ABLATION_CONFIGS = {
    "full": {
        "paper_label": "Search2Adapt",
        "description": "Complete Search2Adapt pipeline (paper baseline)",
        "enable_refinement": True,
        "enable_mcts_search": True,
        "diversity_weight": 0.1,
        "enable_cot_reflection": True,
        "enable_perceptor_context": True,
        "toolset_mode": "full",
    },
    "wo_perceptor": {
        "paper_label": "Search2Adapt w/o Perceptor",
        "description": "Remove the Perceptor: no domain profile in planning/evaluation",
        "enable_refinement": True,
        "enable_mcts_search": True,
        "diversity_weight": 0.1,
        "enable_cot_reflection": True,
        "enable_perceptor_context": False,
        "toolset_mode": "full",
    },
    "wo_cots": {
        "paper_label": "Search2Adapt w/o CoTS",
        "description": (
            "Bypass iterative refinement: keep only the initial pseudo-label "
            "produced by the VFM plan"
        ),
        "enable_refinement": False,
        "enable_mcts_search": True,
        "diversity_weight": 0.1,
        "enable_cot_reflection": True,
        "enable_perceptor_context": True,
        "toolset_mode": "full",
    },
    "wo_cot_diversity": {
        "paper_label": "Search2Adapt w/o CoT Diversity",
        "description": "Remove the diversity bonus from the action-value function (lambda=0)",
        "enable_refinement": True,
        "enable_mcts_search": True,
        "diversity_weight": 0.0,
        "enable_cot_reflection": True,
        "enable_perceptor_context": True,
        "toolset_mode": "full",
    },
    "wo_cot_reflection": {
        "paper_label": "Search2Adapt w/o CoT Reflection",
        "description": "Remove the Strategist-Evaluator CoT reflection step",
        "enable_refinement": True,
        "enable_mcts_search": True,
        "diversity_weight": 0.1,
        "enable_cot_reflection": False,
        "enable_perceptor_context": True,
        "toolset_mode": "full",
    },
    "wo_vfm_postprocess": {
        "paper_label": "Search2Adapt w/o VFM Postprocess",
        "description": "Remove the post-processing operators from the action space",
        "enable_refinement": True,
        "enable_mcts_search": True,
        "diversity_weight": 0.1,
        "enable_cot_reflection": True,
        "enable_perceptor_context": True,
        "toolset_mode": "vfm_only",
    },
}


def apply_ablation_config(
    config: Search2AdaptConfig, ablation_name: str
) -> Search2AdaptConfig:
    """Apply ablation configuration to the Search2AdaptConfig object."""
    if ablation_name not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown ablation config: {ablation_name!r}. "
            f"Available: {', '.join(list_ablation_configs())}"
        )

    preset = ABLATION_CONFIGS[ablation_name]

    config.mcts.enable_refinement = preset["enable_refinement"]
    config.mcts.enable_mcts_search = preset["enable_mcts_search"]
    config.mcts.diversity_weight = preset["diversity_weight"]
    config.mcts.enable_cot_reflection = preset["enable_cot_reflection"]
    config.mcts.enable_perceptor_context = preset["enable_perceptor_context"]
    config.toolbox.toolset_mode = preset["toolset_mode"]

    return config


def list_ablation_configs():
    """Return list of available ablation configuration names."""
    return list(ABLATION_CONFIGS.keys())

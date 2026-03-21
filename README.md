# Search2Adapt: Multi-Agent Collaborative Tree Search for Source-Free Unsupervised Domain Adaptation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Search2Adapt** is a novel multi-agent collaborative framework that reformulates Source-Free Unsupervised Domain Adaptation (SFUDA) from static pseudo-label generation into a dynamic strategy search problem. It introduces CoTS (Collaborative Tree Search), a plan-level search algorithm that enables efficient exploration of pseudo-label refinement strategies through multi-agent collaboration.

## 🔑 Key Features

- **Closed-Loop Multi-Agent Framework**: Transforms traditional open-loop SFUDA pipelines into feedback-driven decision processes
- **Plan-Level Tree Search (CoTS)**: Operates on complete tool-chain plans rather than individual actions, significantly reducing computational overhead
- **Diversity-Augmented UCT**: Prevents premature convergence to suboptimal strategies
- **Anatomy-Aware Reward Mechanism**: Multi-dimensional scoring without ground-truth labels
- **Dynamic Prior Memory**: Accumulates successful strategies across samples for accelerated adaptation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Search2Adapt Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Domain Perception & Planning                          │
│  ┌──────────┐     ┌─────────────┐                              │
│  │Perceptor │────▶│ Strategist  │                              │
│  │          │     │             │                              │
│  └──────────┘     └─────────────┘                              │
│       │                  │                                      │
│       ▼                  ▼                                      │
│  Domain Profile    Adaptation Plans                             │
│                                                                 │
│  Phase 2: Collaborative Tree Search (CoTS)                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    MCTS Search Loop                      │   │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────┐       │   │
│  │  │Strategist │◀──▶│ Evaluator │◀──▶│ Executor  │       │   │
│  │  │ (Planner) │    │  (Critic) │    │  (Actor)  │       │   │
│  │  └───────────┘    └───────────┘    └───────────┘       │   │
│  │        │                │               │                │   │
│  │        ▼                ▼               ▼                │   │
│  │   Plan Proposal    Reward Signal    Tool Execution       │   │
│  │        │                │               │                │   │
│  │        └────────────────┴───────────────┘                │   │
│  │                     │                                    │   │
│  │              CoT Reflection                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│                   Best Pseudo-Labels                            │
│                                                                 │
│  Phase 3: Knowledge Distillation                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              nnUNet Training Pipeline                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│                  Deployable Student Model                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Search2Adapt/
├── search2adapt/
│   ├── agents/
│   │   ├── perceptor.py      # Domain perception agent (VLM-based)
│   │   ├── strategist.py     # Planning agent (plan proposal + reflection)
│   │   ├── evaluator.py      # Critic agent (multi-dimensional scoring)
│   │   ├── executor.py       # Tool execution agent
│   │   └── base.py           # Base classes and utilities
│   ├── mcts/
│   │   ├── tree.py           # PlanNode/PlanTree with diversity-augmented UCT
│   │   └── search.py         # Two-phase search implementation
│   ├── toolbox/
│   │   └── tools.py          # VFM and post-processing tools
│   ├── prompts.py            # LLM/VLM prompt templates
│   ├── memory.py             # Dynamic prior memory
│   ├── config.py             # Configuration management
│   └──  main.py               # Main entry point
│   
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Create conda environment
conda create -n search2adapt python=3.10
conda activate search2adapt

# Install dependencies
pip install -r requirements.txt
```

### Prerequisites

1. **VFM Models** (at least one required):
   - [BiomedParse](https://github.com/microsoft/BiomedParse) for medical image segmentation
   - [SAM3](https://github.com/yx-s-bit/SAM3) for volumetric segmentation

2. **LLM/VLM API**: Configure your API endpoint in `config.py`

3. **nnUNet v2**: For knowledge distillation phase

### Basic Usage

```python
from search2adapt.main import run_full_pipeline

# Run the full pipeline
result = run_full_pipeline(
    target_data_path="/path/to/nnUNet_dataset",
    target_organs=["liver", "spleen", "kidney"],
    result_base_dir="./S2A_Results",
    grid_size=3,
    num_cases=5,
    mcts_num_cases=-1,  # -1 for all cases
    initial_method="biomedparse",
    skip_distillation=False,
)
```

### Ablation Experiments

```python
from search2adapt.main import run_full_pipeline

# Run with ablation configuration
result = run_full_pipeline(
    target_data_path="/path/to/dataset",
    target_organs=["liver", "spleen"]
)
```

## 🔬 Core Innovations

### 1. Closed-Loop Multi-Agent Collaboration

Unlike traditional SFUDA pipelines that apply fixed tool sequences, Search2Adapt introduces a closed-loop decision process where Strategist and Evaluator engage in constructive deliberation:

```
Traditional SFUDA (Open-Loop):
  VFM → Post-process → Pseudo-label → Done

Search2Adapt (Closed-Loop):
  Perceptor → Strategist ⇄ Evaluator → MCTS Search → Reflection → Iterate
```

### 2. Plan-Level Tree Search (CoTS)

CoTS operates on complete tool-chain plans rather than individual actions:

| Phase | Description |
|-------|-------------|
| **Phase 1** | LLM-based plan evaluation (no execution) |
| **Phase 2** | Top-K plans actual execution verification |

### 3. Diversity-Augmented UCT

```
UCT(s_i) = V(s_i) + w·√(ln N_parent/N_i) + λ/√(count(tool_sig_i))
           ─────────   ────────────────────   ────────────────────
           Exploitation  Exploration          Diversity Bonus
```

### 4. Anatomy-Aware Reward Mechanism

Multi-dimensional scoring without ground-truth labels:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Anatomical Plausibility | 0.35 | Organ location, size, shape correctness |
| Topological Integrity | 0.25 | Connected components, fragmentation |
| Boundary Smoothness | 0.20 | Boundary quality, no jagged artifacts |
| Volumetric Consistency | 0.20 | Volume ratios for paired organs |

### 5. Dynamic Prior Memory

Five-layer memory structure for cross-sample knowledge transfer:

- `domain_profile`: Perceptor's domain analysis
- `adaptation_plans`: Strategist's candidate plans
- `trajectories`: Successful tool chains
- `plan_dialogues`: CoTS collaboration records
- `eval_records`: Evaluation history

## 🛠️ Toolbox

Search2Adapt integrates various tools organized by category:

| Category | Tools |
|----------|-------|
| **VFM Proposers** | BiomedParse, SAM3, Dual Expert |
| **Preprocessing** | CLAHE, normalize_intensity, clip_intensity |
| **Post-processing** | morphological_closing/opening, gaussian_smooth_mask, remove_small_labels |

## ⚙️ Configuration

Key configuration parameters in `config.py`:

```python
@dataclass
class MCTSConfig:
    exploration_weight: float = 1.414  # UCT exploration coefficient
    diversity_weight: float = 0.1      # Diversity bonus coefficient
    max_depth: int = 6                 # Maximum tree depth
    max_iterations: int = 50           # Maximum search iterations
    enable_cot_reflection: bool = True # Enable CoT reflection
    enable_mcts_search: bool = True    # Enable MCTS (False = Greedy)
    enable_perceptor_context: bool = True  # Enable Perceptor context

@dataclass
class ToolboxConfig:
    toolset_mode: str = "full"  # "vfm_only", "vfm_postprocess", "full"
```

## 📝 Citation

If you find this work useful, please cite:

```bibtex

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BiomedParse](https://github.com/microsoft/BiomedParse) for vision foundation model
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet) for knowledge distillation framework
- [MONAI](https://monai.io/) for medical image processing tools


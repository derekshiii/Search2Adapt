#!/usr/bin/env python
"""
Minimal end-to-end Search2Adapt example.

Adapts a source-free model to an unlabeled target CT dataset of abdominal
organs (AMOS-style), runs the multi-agent search, and evaluates the resulting
pseudo-labels against the ground truth.

Prerequisites
-------------
1. A target dataset in nnU-Net layout::

       Dataset500_AMOS_CT/
       ├── imagesTr/AMOS_0001_0000.nii.gz ...
       ├── labelsTr/AMOS_0001.nii.gz ...
       └── dataset.json

2. Environment variables (see README.md, Configuration)::

       export SEARCH2ADAPT_LLM_API_KEY=...
       export SEARCH2ADAPT_LLM_BASE_URL=http://localhost:8000/v1
       export SEARCH2ADAPT_BIOMEDPARSE_ROOT=/path/to/BiomedParse
       export SEARCH2ADAPT_BIOMEDPARSE_CHECKPOINT=/path/to/biomedparse_3D.ckpt
       export SEARCH2ADAPT_BIOMEDPARSE_PYTHON=/path/to/envs/biomedparse/bin/python

Usage
-----
    python examples/run_amos_mr_to_ct.py \\
        --dataset /path/to/Dataset500_AMOS_CT \\
        --organs liver,spleen,kidney_left,kidney_right

    # smoke test: search only the first 3 cases, skip distillation
    python examples/run_amos_mr_to_ct.py --dataset ... --smoke-test
"""

import argparse
import os
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Search2Adapt — MR→CT adaptation on AMOS-style abdominal data."
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="nnU-Net target dataset directory (contains imagesTr/)",
    )
    p.add_argument(
        "--organs",
        default="liver,spleen,kidney_left,kidney_right",
        help="comma-separated organ names used as VFM text prompts",
    )
    p.add_argument(
        "--result-dir",
        default="./S2A_Results",
        help="parent directory for run outputs (default: ./S2A_Results)",
    )
    p.add_argument(
        "--initial-method",
        default="biomedparse",
        choices=["biomedparse", "sam3", "dual_expert"],
        help="VFM used for the initial pseudo-label (default: biomedparse)",
    )
    p.add_argument(
        "--ablation",
        default="full",
        help="ablation preset; run `search2adapt list-ablations` to see them",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="search 3 cases only and skip distillation",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    organs = [o.strip() for o in args.organs.split(",") if o.strip()]
    if not organs:
        print("error: --organs must name at least one organ", file=sys.stderr)
        return 1

    dataset = os.path.abspath(os.path.expanduser(args.dataset))
    images_tr = os.path.join(dataset, "imagesTr")
    if not os.path.isdir(images_tr):
        print(
            f"error: {images_tr} not found — the dataset must use the nnU-Net "
            f"layout (see docs/DATASETS.md)",
            file=sys.stderr,
        )
        return 1

    # Import here so `--help` works even without the runtime dependencies.
    from search2adapt.main import run_full_pipeline

    result = run_full_pipeline(
        target_data_path=dataset,
        target_organs=organs,
        result_base_dir=args.result_dir,
        grid_size=3,
        num_cases=5,
        mcts_num_cases=3 if args.smoke_test else -1,
        initial_method=args.initial_method,
        ablation_name=args.ablation,
        num_plans=4,
        skip_distillation=args.smoke_test,
    )

    result_dir = result["result_dir"]
    print(f"\nPseudo-labels written to: {os.path.join(result_dir, 'final_pseudolabels')}")

    labels_tr = os.path.join(dataset, "labelsTr")
    if os.path.isdir(labels_tr):
        from search2adapt.metrics import evaluate_result_dir, print_metrics_table

        print("\nEvaluating pseudo-labels against ground truth:\n")
        metrics = evaluate_result_dir(
            result_dir=result_dir,
            gt_dir=labels_tr,
            label_names={i + 1: name for i, name in enumerate(organs)},
        )
        if "error" in metrics:
            print(f"evaluation failed: {metrics['error']}", file=sys.stderr)
            return 1
        print_metrics_table(metrics)
    else:
        print(f"\nNo labelsTr/ in {dataset} — skipping the evaluation step.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

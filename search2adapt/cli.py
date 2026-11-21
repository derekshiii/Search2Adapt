"""
Search2Adapt command-line interface.

Examples:
    # full pipeline on an nnU-Net target dataset
    search2adapt run \\
        --dataset /path/to/DatasetXXX_AMOS_CT \\
        --organs "spleen,liver,aorta" \\
        --result-dir ./S2A_Results

    # an ablation row of Table 5
    search2adapt run --dataset ... --organs ... --ablation wo_cot_diversity

    # domain analysis only
    search2adapt perceptor --dataset ... --organs ...

    # Dice / ASD against ground truth
    search2adapt evaluate --result-dir ... --gt-dir ...

    search2adapt list-ablations
"""

import argparse
import json
import os
import sys
from typing import List, Optional

from search2adapt.ablation_configs import ABLATION_CONFIGS, list_ablation_configs


def _split_organs(value: str) -> List[str]:
    organs = [o.strip() for o in value.replace(";", ",").split(",")]
    return [o for o in organs if o]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search2adapt",
        description="Search2Adapt: multi-agent collaborative tree search for "
        "source-free unsupervised domain adaptation in medical image segmentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the full pipeline (phases 1-3)")
    run_p.add_argument(
        "--dataset",
        required=True,
        help="nnU-Net target dataset directory (contains imagesTr/)",
    )
    run_p.add_argument(
        "--organs",
        required=True,
        help="comma-separated target organ names used as VFM text prompts",
    )
    run_p.add_argument(
        "--result-dir",
        default="./S2A_Results",
        help="parent directory for run outputs (default: ./S2A_Results)",
    )
    run_p.add_argument(
        "--initial-method",
        default="biomedparse",
        choices=["biomedparse", "sam3", "dual_expert"],
        help="VFM used to generate the initial pseudo-label (default: biomedparse)",
    )
    run_p.add_argument(
        "--ablation",
        default="full",
        choices=list_ablation_configs(),
        help="ablation preset (default: full)",
    )
    run_p.add_argument(
        "--skip-distillation",
        action="store_true",
        help="stop after pseudo-label optimization (skip nnU-Net training)",
    )
    run_p.add_argument(
        "--mcts-num-cases",
        type=int,
        default=-1,
        help="number of cases to run the search on; -1 = all (default: -1)",
    )
    run_p.add_argument(
        "--num-plans",
        type=int,
        default=4,
        help="candidate adaptation plans proposed per expansion (default: 4)",
    )
    run_p.add_argument(
        "--grid-size",
        type=int,
        default=3,
        help="N x N slice grid shown to the VLM during domain analysis (default: 3)",
    )
    run_p.add_argument(
        "--perceptor-cases",
        type=int,
        default=5,
        help="cases sampled for the Perceptor domain analysis (default: 5)",
    )
    run_p.add_argument(
        "--min-reward",
        type=float,
        default=0.0,
        help="minimum pseudo-label reward required for distillation (default: 0.0)",
    )
    run_p.add_argument(
        "--cleanup-temp",
        action="store_true",
        help="delete the temporary grid images after the Perceptor stage",
    )

    perc_p = sub.add_parser(
        "perceptor", help="run the Perceptor domain analysis only"
    )
    perc_p.add_argument("--dataset", required=True, help="target dataset directory")
    perc_p.add_argument("--organs", required=True, help="comma-separated organ names")
    perc_p.add_argument(
        "--result-dir",
        default="./S2A_Results",
        help="parent directory for run outputs (default: ./S2A_Results)",
    )
    perc_p.add_argument(
        "--grid-size", type=int, default=3, help="N x N slice grid size (default: 3)"
    )
    perc_p.add_argument(
        "--num-cases", type=int, default=5, help="cases sampled for analysis (default: 5)"
    )

    eval_p = sub.add_parser(
        "evaluate", help="compute Dice / ASD against ground-truth labels"
    )
    eval_p.add_argument(
        "--result-dir", required=True, help="run directory containing final_pseudolabels/"
    )
    eval_p.add_argument(
        "--gt-dir", required=True, help="directory containing ground-truth NIfTI files"
    )
    eval_p.add_argument(
        "--label-ids",
        default="",
        help="comma-separated label IDs to evaluate (default: all non-zero)",
    )
    eval_p.add_argument(
        "--label-names",
        default="",
        help="comma-separated label names, positional to --label-ids",
    )
    eval_p.add_argument(
        "--compute-hd",
        action="store_true",
        help="also compute HD95 (slow)",
    )

    sub.add_parser("list-ablations", help="list the available ablation presets")

    return parser


def _cmd_run(args) -> int:
    from search2adapt.main import run_full_pipeline

    organs = _split_organs(args.organs)
    if not organs:
        print("error: --organs must list at least one organ", file=sys.stderr)
        return 2

    result = run_full_pipeline(
        target_data_path=os.path.abspath(os.path.expanduser(args.dataset)),
        target_organs=organs,
        result_base_dir=os.path.abspath(os.path.expanduser(args.result_dir)),
        grid_size=args.grid_size,
        num_cases=args.perceptor_cases,
        mcts_num_cases=args.mcts_num_cases,
        cleanup_temp=args.cleanup_temp,
        initial_method=args.initial_method,
        skip_distillation=args.skip_distillation,
        min_reward=args.min_reward,
        num_plans=args.num_plans,
        ablation_name=args.ablation,
    )
    print(json.dumps({"result_dir": result["result_dir"]}, indent=2))
    return 0


def _cmd_perceptor(args) -> int:
    from search2adapt.main import run_perceptor_only

    organs = _split_organs(args.organs)
    if not organs:
        print("error: --organs must list at least one organ", file=sys.stderr)
        return 2

    plan = run_perceptor_only(
        target_data_path=os.path.abspath(os.path.expanduser(args.dataset)),
        target_organs=organs,
        result_base_dir=os.path.abspath(os.path.expanduser(args.result_dir)),
        grid_size=args.grid_size,
        num_cases=args.num_cases,
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_evaluate(args) -> int:
    from search2adapt.metrics import evaluate_result_dir, print_metrics_table

    label_ids = None
    if args.label_ids.strip():
        label_ids = [int(x) for x in args.label_ids.split(",") if x.strip()]

    label_names = None
    if args.label_names.strip():
        names = [x.strip() for x in args.label_names.split(",") if x.strip()]
        if label_ids is None:
            label_ids = list(range(1, len(names) + 1))
        label_names = {lid: name for lid, name in zip(label_ids, names)}

    metrics = evaluate_result_dir(
        result_dir=os.path.abspath(os.path.expanduser(args.result_dir)),
        gt_dir=os.path.abspath(os.path.expanduser(args.gt_dir)),
        label_ids=label_ids,
        label_names=label_names,
        compute_hd=args.compute_hd,
    )
    if "error" in metrics:
        print(f"error: {metrics['error']}", file=sys.stderr)
        return 1
    print_metrics_table(metrics)
    return 0


def _cmd_list_ablations(args) -> int:
    width = max(len(k) for k in ABLATION_CONFIGS)
    for name, preset in ABLATION_CONFIGS.items():
        print(f"{name.ljust(width)}  {preset['paper_label']}")
        print(f"{' ' * width}  {preset['description']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "run": _cmd_run,
        "perceptor": _cmd_perceptor,
        "evaluate": _cmd_evaluate,
        "list-ablations": _cmd_list_ablations,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

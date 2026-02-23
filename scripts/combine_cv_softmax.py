"""
Combine 5-fold cross-validation softmax .npz files into a single directory per task.

In 5-fold CV each case appears in exactly one fold's validation_raw, so combining
simply collects all .npz files across folds (no averaging needed).

Creates symlinks to avoid copying large data.

Usage:
    uv run python scripts/combine_cv_softmax.py \
        --results_base /path/to/nnUNet_results/nnUNet/3d_fullres \
        --tasks 611 612 613 614 \
        --trainer nnUNetTrainerV2_fast \
        --plans nnUNetPlansv2.1 \
        --output_base /path/to/output/cv_softmax
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_base", required=True,
                   help="Base directory containing TaskXXX_* folders.")
    p.add_argument("--tasks", nargs="+", type=int, default=[611, 612, 613, 614])
    p.add_argument("--trainer", default="nnUNetTrainerV2_fast")
    p.add_argument("--plans", default="nnUNetPlansv2.1")
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output_base", required=True,
                   help="Output base dir. Per-task dirs created as <output_base>/Task<id>/.")
    return p.parse_args()


def main():
    args = parse_args()
    results_base = Path(args.results_base)
    output_base = Path(args.output_base)

    for task_id in args.tasks:
        # Locate the task directory (e.g. Task611_TotalSegGroup1)
        task_dirs = sorted(results_base.glob(f"Task{task_id}_*"))
        if not task_dirs:
            print(f"[ERROR] No directory found for Task{task_id} in {results_base}", file=sys.stderr)
            continue
        task_dir = task_dirs[0] / f"{args.trainer}__{args.plans}"

        out_dir = output_base / f"Task{task_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        total, skipped = 0, 0
        for fold in args.folds:
            fold_dir = task_dir / f"fold_{fold}" / "validation_raw"
            if not fold_dir.exists():
                print(f"  [WARN] Task{task_id} fold_{fold}: {fold_dir} not found", file=sys.stderr)
                continue

            npz_files = sorted(fold_dir.glob("*.npz"))
            if not npz_files:
                print(f"  [WARN] Task{task_id} fold_{fold}: no .npz files found", file=sys.stderr)
                continue

            for src in npz_files:
                dst = out_dir / src.name
                if dst.exists() or dst.is_symlink():
                    # Duplicate case across folds — should not happen in proper 5-fold CV
                    print(f"  [WARN] Duplicate case {src.name} (fold_{fold}), skipping", file=sys.stderr)
                    skipped += 1
                    continue
                dst.symlink_to(src.resolve())
                total += 1

        print(f"Task{task_id}: {total} cases combined into {out_dir}"
              + (f" ({skipped} skipped duplicates)" if skipped else ""))


if __name__ == "__main__":
    main()

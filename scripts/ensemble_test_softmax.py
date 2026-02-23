"""
Average softmax .npz predictions across 5 folds for each task.

After nnUNet_predict is run for all 5 folds, this script averages the
per-fold softmax arrays per case and saves the result as a single .npz.

Usage:
    uv run python scripts/ensemble_test_softmax.py \
        --pred_base /scratch/work/tianmid1/data/test_predictions \
        --tasks 611 612 613 614 \
        --folds 0 1 2 3 4 \
        --output_base /scratch/work/tianmid1/data/test_ensemble
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_base", required=True,
                   help="Base dir with Task<id>/fold_<n>/ subdirs.")
    p.add_argument("--tasks", nargs="+", type=int, default=[611, 612, 613, 614])
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output_base", required=True,
                   help="Output base dir. Per-task dirs created as <output_base>/Task<id>/.")
    return p.parse_args()


def ensemble_task(task_id, pred_base, folds, out_dir):
    fold_dirs = []
    for f in folds:
        fd = pred_base / f"Task{task_id}" / f"fold_{f}"
        if not fd.exists():
            print(f"  [WARN] Task{task_id} fold_{f}: {fd} not found, skipping", file=sys.stderr)
            continue
        fold_dirs.append(fd)

    if not fold_dirs:
        print(f"[ERROR] Task{task_id}: no fold directories found", file=sys.stderr)
        return

    # Collect case IDs from first available fold
    case_ids = sorted(p.stem for p in fold_dirs[0].glob("*.npz"))
    if not case_ids:
        print(f"[ERROR] Task{task_id}: no .npz files found in {fold_dirs[0]}", file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    n_missing, n_done = 0, 0

    for case_id in case_ids:
        softmax_list = []
        for fd in fold_dirs:
            npz_path = fd / f"{case_id}.npz"
            if not npz_path.exists():
                print(f"  [WARN] Task{task_id} {case_id}: missing in {fd.name}", file=sys.stderr)
                continue
            softmax_list.append(np.load(str(npz_path))["softmax"].astype(np.float32))

        if not softmax_list:
            print(f"  [WARN] Task{task_id} {case_id}: no predictions found", file=sys.stderr)
            n_missing += 1
            continue

        averaged = np.mean(softmax_list, axis=0)  # [K+1, D, H, W]
        np.savez_compressed(str(out_dir / f"{case_id}.npz"), softmax=averaged)
        n_done += 1

    print(f"Task{task_id}: {n_done} cases ensembled from {len(fold_dirs)} folds → {out_dir}"
          + (f" ({n_missing} missing)" if n_missing else ""))


def main():
    args = parse_args()
    pred_base = Path(args.pred_base)
    output_base = Path(args.output_base)

    for task_id in args.tasks:
        out_dir = output_base / f"Task{task_id}"
        ensemble_task(task_id, pred_base, args.folds, out_dir)


if __name__ == "__main__":
    main()

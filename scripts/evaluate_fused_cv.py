"""
Evaluate all 5-fold fused predictions against GT and produce a single
cross-validation summary.json (equivalent to nnUNet's cv_niftis_raw/summary.json).

Each case appears in exactly one fold, so pooling all folds gives the full
cross-validation result without any duplicates.

Usage:
    uv run python scripts/evaluate_fused_cv.py \
        --fused_dir /path/to/fused_grouped \
        --gt_dir    /path/to/gt_segmentations \
        --output    /path/to/fused_grouped/cv_summary.json \
        --num_threads 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.evaluation.evaluator import aggregate_scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fused_dir", required=True,
                   help="Root folder containing fold_0 … fold_4 subdirectories")
    p.add_argument("--gt_dir", required=True,
                   help="Folder with GT segmentations (TotalSegmentator_XXXX.nii.gz)")
    p.add_argument("--output", required=True,
                   help="Output path for the combined summary.json")
    p.add_argument("--folds", nargs="+", default=["fold_0","fold_1","fold_2","fold_3","fold_4"],
                   help="Fold subdirectory names to include (default: all 5)")
    p.add_argument("--num_threads", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    fused_dir = Path(args.fused_dir)
    gt_dir = Path(args.gt_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    test_ref_pairs = []
    missing_gt = []

    for fold in args.folds:
        fold_dir = fused_dir / fold
        if not fold_dir.exists():
            print(f"[WARN] Fold directory not found: {fold_dir}", file=sys.stderr)
            continue
        pred_files = sorted(fold_dir.glob("TotalSeg_*.nii.gz"))
        for pf in pred_files:
            case_num = pf.stem.replace(".nii", "").replace("TotalSeg_", "")
            gt_file = gt_dir / f"TotalSegmentator_{case_num}.nii.gz"
            if not gt_file.exists():
                missing_gt.append(pf.name)
                continue
            test_ref_pairs.append((str(pf), str(gt_file)))

    if missing_gt:
        print(f"[WARN] No GT found for {len(missing_gt)} cases: "
              f"{missing_gt[:5]}{'...' if len(missing_gt) > 5 else ''}", file=sys.stderr)

    if not test_ref_pairs:
        print("ERROR: No valid (pred, gt) pairs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Folds   : {args.folds}")
    print(f"Cases   : {len(test_ref_pairs)}")
    print(f"Labels  : 0-104 (105 total)")
    print(f"Output  : {output}")
    print(f"Threads : {args.num_threads}")
    print()

    labels = list(range(105))  # 0=background, 1-104=organs

    aggregate_scores(
        test_ref_pairs,
        labels=labels,
        json_output_file=str(output),
        json_name="fused_grouped_cv",
        json_description="5-fold CV: fused Task611-614 group predictions (global labels 0-104)",
        json_author="auto",
        json_task="fused_grouped_cv",
        num_threads=args.num_threads,
    )

    print(f"\nDone. Summary written to: {output}")


if __name__ == "__main__":
    main()

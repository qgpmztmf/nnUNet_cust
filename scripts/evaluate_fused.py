"""
Generate summary.json for fused group predictions, matching nnUNet's format.

Fused predictions: TotalSeg_XXXX.nii.gz  (global labels 0-104)
GT segmentations:  TotalSegmentator_XXXX.nii.gz  (global labels 0-104)

Usage:
    uv run python scripts/evaluate_fused.py \
        --pred_dir  /path/to/fused_grouped/fold_0 \
        --gt_dir    /path/to/nnUNet_preprocessed/Task601_TotalSegmentatorV1/gt_segmentations \
        --output    /path/to/fused_grouped/fold_0/summary.json \
        --num_threads 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.evaluation.evaluator import aggregate_scores


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True,
                   help="Folder with fused predictions (TotalSeg_XXXX.nii.gz)")
    p.add_argument("--gt_dir", required=True,
                   help="Folder with GT segmentations (TotalSegmentator_XXXX.nii.gz)")
    p.add_argument("--output", required=True,
                   help="Output path for summary.json")
    p.add_argument("--num_threads", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("TotalSeg_*.nii.gz"))
    if not pred_files:
        print(f"ERROR: No TotalSeg_*.nii.gz files found in {pred_dir}", file=sys.stderr)
        sys.exit(1)

    # Build (pred, gt) pairs — TotalSeg_0000 -> TotalSegmentator_0000
    test_ref_pairs = []
    missing = []
    for pf in pred_files:
        case_num = pf.stem.replace(".nii", "").replace("TotalSeg_", "")
        gt_file = gt_dir / f"TotalSegmentator_{case_num}.nii.gz"
        if not gt_file.exists():
            missing.append(str(pf.name))
            continue
        test_ref_pairs.append((str(pf), str(gt_file)))

    if missing:
        print(f"[WARN] No GT found for {len(missing)} cases: {missing[:5]}{'...' if len(missing)>5 else ''}",
              file=sys.stderr)

    print(f"Evaluating {len(test_ref_pairs)} cases with 105 labels (0-104) ...")

    labels = list(range(105))  # 0=background, 1-104=organs

    aggregate_scores(
        test_ref_pairs,
        labels=labels,
        json_output_file=str(output),
        json_name="fused_grouped val",
        json_description="Fused Task611-614 group predictions",
        json_author="auto",
        json_task="fused_grouped",
        num_threads=args.num_threads,
    )

    print(f"Done. Summary written to: {output}")


if __name__ == "__main__":
    main()

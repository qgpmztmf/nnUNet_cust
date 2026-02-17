"""
Fuse predictions from 4 group-specific nnUNet models into a single 104-class segmentation.

Each model outputs softmax probabilities for its own local label space.
This script:
  1. Maps each model's local class probabilities to global 104-class IDs
  2. Builds a combined probability volume [105, D, H, W] (background + 104 classes)
  3. For background: uses the minimum background prob across all models
     (if ANY model thinks it's not background, we trust that)
  4. Takes the global argmax per voxel
  5. Applies priority rule for tie-breaking: bone(4) > vascular(3) > soft_tissue(2) > lung_GI(1)

Usage:
    python fuse_group_predictions.py \
        --pred_dirs  /path/to/task611/predictions \
                     /path/to/task612/predictions \
                     /path/to/task613/predictions \
                     /path/to/task614/predictions \
        --mapping_files /path/to/Task611/class_mapping.json \
                        /path/to/Task612/class_mapping.json \
                        /path/to/Task613/class_mapping.json \
                        /path/to/Task614/class_mapping.json \
        --output_dir /path/to/fused_predictions \
        --case_ids TotalSeg_0001 TotalSeg_0002  (optional, default: all)

It can also be used as a library:
    from fuse_group_predictions import GroupFuser
    fuser = GroupFuser(mapping_files=[...])
    fused_label = fuser.fuse_softmax(softmax_dict)  # {group_id: np.array[K+1, D, H, W]}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# GroupFuser: core fusion logic (usable as library)
# ---------------------------------------------------------------------------

class GroupFuser:
    """Fuse softmax predictions from multiple group-specific models."""

    def __init__(self, mapping_files: List[str]):
        """
        Args:
            mapping_files: list of class_mapping.json paths, one per group model.
        """
        self.mappings = {}       # {group_id: mapping_dict}
        self.local_to_global = {}  # {group_id: {local_id: global_id}}
        self.global_to_group = {}  # {global_id: group_id}
        self.priorities = {}     # {group_id: priority}
        self.n_global_classes = 104  # TotalSegmentator

        for mf in mapping_files:
            with open(mf) as f:
                m = json.load(f)
            gid = m["group_id"]
            self.mappings[gid] = m
            self.priorities[gid] = m["priority"]
            self.local_to_global[gid] = {
                int(local_id): info["global_id"]
                for local_id, info in m["local_to_global"].items()
            }
            for local_id, info in m["local_to_global"].items():
                self.global_to_group[info["global_id"]] = gid

        # Verify all 104 classes are covered
        covered = set(self.global_to_group.keys())
        expected = set(range(1, 105))
        if covered != expected:
            missing = expected - covered
            extra = covered - expected
            print(f"[WARN] Class coverage: missing={missing}, extra={extra}", file=sys.stderr)

    def fuse_softmax(self,
                     softmax_dict: Dict[int, np.ndarray],
                     ) -> np.ndarray:
        """Fuse softmax probability arrays from multiple group models.

        Args:
            softmax_dict: {group_id: ndarray of shape [K_g+1, D, H, W]}
                          where channel 0 is background, channels 1..K_g are local classes.

        Returns:
            fused_labels: ndarray of shape [D, H, W] with global label IDs (0..104).
        """
        # Determine spatial shape from the first model's output
        first_key = next(iter(softmax_dict))
        spatial_shape = softmax_dict[first_key].shape[1:]

        # Global probability tensor: [105, D, H, W]
        # Channel 0 = background, channels 1..104 = global class IDs
        global_probs = np.zeros((self.n_global_classes + 1,) + spatial_shape, dtype=np.float32)

        # Track background: use minimum across all models
        # (if any model is confident it's NOT background, trust that)
        bg_probs = []

        for gid, softmax in softmax_dict.items():
            if gid not in self.local_to_global:
                print(f"[WARN] Unknown group {gid}, skipping", file=sys.stderr)
                continue

            # Collect background probability
            bg_probs.append(softmax[0])

            # Map local class probs to global positions
            for local_id, global_id in self.local_to_global[gid].items():
                global_probs[global_id] = softmax[local_id]

        # Background = minimum across all models
        if bg_probs:
            global_probs[0] = np.minimum.reduce(bg_probs)

        # Argmax with priority-based tie-breaking
        # Add tiny epsilon scaled by priority to break ties in favor of higher-priority groups
        priority_bonus = np.zeros_like(global_probs)
        for gid, l2g in self.local_to_global.items():
            eps = self.priorities[gid] * 1e-7
            for local_id, global_id in l2g.items():
                priority_bonus[global_id] = eps

        fused_labels = np.argmax(global_probs + priority_bonus, axis=0).astype(np.uint8)
        return fused_labels

    def fuse_argmax(self,
                    label_dict: Dict[int, np.ndarray],
                    ) -> np.ndarray:
        """Simpler fusion from hard label predictions (no softmax).

        Uses priority rule directly: for each voxel, among models predicting
        non-background, the highest-priority group wins.

        Args:
            label_dict: {group_id: ndarray of shape [D, H, W] with local label IDs}

        Returns:
            fused_labels: ndarray of shape [D, H, W] with global label IDs (0..104).
        """
        first_key = next(iter(label_dict))
        spatial_shape = label_dict[first_key].shape
        fused = np.zeros(spatial_shape, dtype=np.uint8)

        # Apply groups in priority order (lowest first, highest overwrites)
        for gid in sorted(label_dict.keys(), key=lambda g: self.priorities.get(g, 0)):
            local_labels = label_dict[gid]
            l2g = self.local_to_global[gid]

            for local_id, global_id in l2g.items():
                mask = local_labels == local_id
                fused[mask] = global_id

        return fused

    def get_group_info(self) -> str:
        """Return a human-readable summary of group configuration."""
        lines = []
        for gid in sorted(self.mappings.keys()):
            m = self.mappings[gid]
            lines.append(f"Group {gid} ({m['group_name']}): "
                         f"Task{m['task_id']}, {m['num_classes']} classes, "
                         f"priority={m['priority']}, "
                         f"HU [{m['hu_window']['low']:.1f}, {m['hu_window']['high']:.1f}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# File-based fusion (reads nnUNet prediction NIfTIs)
# ---------------------------------------------------------------------------

def fuse_case(fuser: GroupFuser,
              case_name: str,
              pred_dirs: Dict[int, Path],
              output_dir: Path,
              use_softmax: bool = False) -> None:
    """Fuse predictions for a single case from nnUNet output directories.

    nnUNet saves hard labels as .nii.gz by default. If use_softmax=True,
    looks for softmax .npz files instead (from --save_softmax flag).
    """
    if use_softmax:
        softmax_dict = {}
        for gid, pdir in pred_dirs.items():
            npz_path = pdir / f"{case_name}.npz"
            if not npz_path.exists():
                print(f"  [WARN] Missing {npz_path}, skipping group {gid}", file=sys.stderr)
                continue
            data = np.load(str(npz_path))
            softmax_dict[gid] = data["softmax"]
        if not softmax_dict:
            print(f"  [ERROR] No predictions found for {case_name}", file=sys.stderr)
            return
        fused_array = fuser.fuse_softmax(softmax_dict)
    else:
        label_dict = {}
        for gid, pdir in pred_dirs.items():
            nii_path = pdir / f"{case_name}.nii.gz"
            if not nii_path.exists():
                print(f"  [WARN] Missing {nii_path}, skipping group {gid}", file=sys.stderr)
                continue
            img = sitk.ReadImage(str(nii_path))
            label_dict[gid] = sitk.GetArrayFromImage(img).astype(np.uint8)
            ref_img = img  # keep for spatial metadata
        if not label_dict:
            print(f"  [ERROR] No predictions found for {case_name}", file=sys.stderr)
            return
        fused_array = fuser.fuse_argmax(label_dict)

    # Write output
    fused_img = sitk.GetImageFromArray(fused_array)
    fused_img.CopyInformation(ref_img)
    output_path = output_dir / f"{case_name}.nii.gz"
    sitk.WriteImage(fused_img, str(output_path), useCompression=True)


def fuse_all(mapping_files: List[str],
             pred_dirs: List[str],
             output_dir: str,
             case_ids: Optional[List[str]] = None,
             use_softmax: bool = False) -> None:
    """Fuse predictions for all cases."""

    fuser = GroupFuser(mapping_files)
    print(fuser.get_group_info())
    print()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map group_id -> prediction directory
    pred_dir_map = {}
    for mf, pd in zip(mapping_files, pred_dirs):
        with open(mf) as f:
            gid = json.load(f)["group_id"]
        pred_dir_map[gid] = Path(pd)

    # Discover case IDs from first prediction directory
    if case_ids is None:
        first_dir = next(iter(pred_dir_map.values()))
        if use_softmax:
            case_ids = sorted([p.stem for p in first_dir.glob("*.npz")])
        else:
            case_ids = sorted([p.stem.replace(".nii", "")
                               for p in first_dir.glob("*.nii.gz")])

    print(f"Fusing {len(case_ids)} cases ...")
    for case_name in case_ids:
        fuse_case(fuser, case_name, pred_dir_map, output_dir, use_softmax)

    print(f"Done. Fused predictions written to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fuse group-specific nnUNet predictions.")
    p.add_argument("--pred_dirs", nargs="+", required=True,
                   help="Prediction directories, one per group (order must match --mapping_files).")
    p.add_argument("--mapping_files", nargs="+", required=True,
                   help="class_mapping.json files, one per group.")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for fused predictions.")
    p.add_argument("--case_ids", nargs="+", default=None,
                   help="Case IDs to process (default: all found in first pred_dir).")
    p.add_argument("--use_softmax", action="store_true",
                   help="Use softmax .npz files instead of hard label .nii.gz files.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fuse_all(
        mapping_files=args.mapping_files,
        pred_dirs=args.pred_dirs,
        output_dir=args.output_dir,
        case_ids=args.case_ids,
        use_softmax=args.use_softmax,
    )

"""
fuse.py — Inference-time ensemble fusion for disjoint-label 3D medical segmentation.

Coordinate spaces
─────────────────
nnUNet_predict saves softmax .npz in INTERNAL space (C, D, H, W) = (C, Z, Y, X).
The hard-label .nii.gz is saved in PATIENT space (correct affine, shape X×Y×Z).
To evaluate against labelsTs, fused predictions must be in patient space.
This script resamples the fused argmax from internal → patient space using
nearest-neighbour zoom, guided by a reference .nii.gz from nnUNet_predict.

Expected folder structure
─────────────────────────
Method A (pre-ensembled per model — 5-fold softmax average already applied):
  <input_root>/
    Task611/<case_id>.npz    # (13, D, H, W) internal space
    Task612/<case_id>.npz    # (25, D, H, W)
    Task613/<case_id>.npz    # (10, D, H, W)
    Task614/<case_id>.npz    # (60, D, H, W)

Method B (raw per-fold softmax, fold averaging done here):
  <per_fold_root>/
    Task611/fold_0/<case_id>.npz  ...  Task614/fold_4/<case_id>.npz

Reference NIfTI (per-fold hard-label, already in patient space):
  <ref_nii_root>/Task611/fold_0/<case_id>.nii.gz   (any task/fold — all share affine)

class_map.json format
─────────────────────
{"611": {"1": 6, "2": 13, ...}, "612": {...}, "613": {...}, "614": {...}}
(auto-built from nnUNet class_mapping.json if --nnunet_raw_data is given)

Example
───────
python fuse.py \\
  --cases_list   test_cases.txt \\
  --input_root   /data/test_ensemble \\
  --ref_nii_root /data/test_predictions \\
  --output_root  /data/test_fused_A \\
  --nnunet_raw_data /data/nnUNet_raw_data \\
  --method A --save_nii
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import zoom

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

ClassMap = Dict[int, int]             # local_id  → global_id
ModelClassMaps = Dict[int, ClassMap]  # task_id   → ClassMap


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_probs(path: Path) -> np.ndarray:
    """Load softmax from .npz (key='softmax') or .npy → float32 (C, D, H, W)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    if path.suffix == ".npz":
        npz = np.load(str(path))
        key = ("softmax" if "softmax" in npz else
               "prob"    if "prob"    in npz else next(iter(npz)))
        arr = npz[key]
    else:
        arr = np.load(str(path))
    return arr.astype(np.float32, copy=False)


def load_class_map_json(path: Path) -> ModelClassMaps:
    with open(path) as f:
        raw = json.load(f)
    return {int(tid): {int(loc): int(glob) for loc, glob in m.items()}
            for tid, m in raw.items()}


def load_class_map_from_nnunet(raw_data_root: Path,
                                task_ids: List[int]) -> ModelClassMaps:
    result: ModelClassMaps = {}
    for tid in task_ids:
        candidates = sorted(raw_data_root.glob(f"Task{tid}_*"))
        if not candidates:
            raise FileNotFoundError(f"No Task{tid}_* under {raw_data_root}")
        with open(candidates[0] / "class_mapping.json") as f:
            d = json.load(f)
        result[tid] = {int(k): int(v["global_id"])
                       for k, v in d["local_to_global"].items()}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Patient-space resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample_seg_to_patient_space(
        seg: np.ndarray,
        ref_nii_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Resample fused label map from nnUNet internal space to patient space.

    nnUNet internal convention : seg is (D, H, W) = (Z, Y, X)
    NiBabel convention         : data stored as (X, Y, Z)

    Steps:
      1. Load reference .nii.gz → get target shape (X, Y, Z) and affine
      2. Convert target shape to (D, H, W) = (Z, Y, X)
      3. Compute per-axis zoom factors: target_DHW / source_DHW
      4. Nearest-neighbour zoom (order=0) — preserves integer labels
      5. Transpose result (D, H, W) → (X, Y, Z) for NiBabel

    Args:
        seg          : (D, H, W) uint16 argmax in nnUNet internal space
        ref_nii_path : per-fold hard-label .nii.gz (patient space, any task/fold)

    Returns:
        resampled_xyz : uint16 array in nibabel (X, Y, Z) order
        affine        : 4×4 affine from the reference image
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required: pip install nibabel")

    ref = nib.load(str(ref_nii_path))
    affine = ref.affine.copy()

    # Reference nibabel shape is (X, Y, Z).
    # Convert to nnUNet (D, H, W) = (Z, Y, X) for comparison with seg.
    ref_xyz_shape = np.asarray(ref.dataobj).shape   # (X, Y, Z)
    target_dhw = (ref_xyz_shape[2], ref_xyz_shape[1], ref_xyz_shape[0])  # (Z, Y, X)

    zoom_factors = tuple(float(t) / float(s)
                         for t, s in zip(target_dhw, seg.shape))

    if all(abs(z - 1.0) < 1e-6 for z in zoom_factors):
        resampled_dhw = seg.copy()
    else:
        resampled_dhw = zoom(
            seg.astype(np.float32), zoom_factors, order=0
        ).astype(np.uint16)

    # (D, H, W) = (Z, Y, X)  →  (X, Y, Z) for nibabel
    resampled_xyz = resampled_dhw.transpose(2, 1, 0)
    return resampled_xyz, affine


def find_ref_nii(case_id: str, ref_nii_root: Path,
                 task_ids: List[int], folds: List[int]) -> Path:
    """Return first available per-fold .nii.gz for this case."""
    for tid in task_ids:
        for f in folds:
            p = ref_nii_root / f"Task{tid}" / f"fold_{f}" / f"{case_id}.nii.gz"
            if p.exists():
                return p
    raise FileNotFoundError(
        f"No reference .nii.gz for {case_id} under {ref_nii_root}")


def save_nii(arr: np.ndarray, affine: np.ndarray, out_path: Path) -> None:
    if not HAS_NIBABEL:
        raise ImportError("nibabel required: pip install nibabel")
    img = nib.Nifti1Image(arr, affine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_probs(probs_by_model: Dict[int, np.ndarray],
                   class_maps: ModelClassMaps) -> Tuple[int, int, int]:
    spatial: Optional[Tuple[int, int, int]] = None
    for task_id, probs in probs_by_model.items():
        if probs.ndim != 4:
            raise ValueError(f"Task{task_id}: expected 4-D (C,D,H,W), got {probs.shape}")
        C, D, H, W = probs.shape
        expected_C = len(class_maps[task_id]) + 1
        if C != expected_C:
            raise ValueError(
                f"Task{task_id}: expected {expected_C} channels, got {C}")
        sp = (D, H, W)
        if spatial is None:
            spatial = sp
        elif sp != spatial:
            raise ValueError(f"Task{task_id}: shape {sp} != reference {spatial}")
    assert spatial is not None
    return spatial


# ─────────────────────────────────────────────────────────────────────────────
# Core fusion (in nnUNet internal space)
# ─────────────────────────────────────────────────────────────────────────────

def fuse_models_disjoint(probs_by_model: Dict[int, np.ndarray],
                         class_maps: ModelClassMaps,
                         eps: float = 1e-8) -> np.ndarray:
    """Fuse per-model softmax into a single 105-channel global probability.

    Fusion rule (per voxel):
      Background : product of all models' bg probs in log-space (numerically stable)
                   s_bg = exp( Σ_m log( clip(p_bg^m, eps, 1) ) )
      Foreground : copy each model's local channel directly to global channel
                   s_c  = p_{m(c)}( local_c )
      Normalize  : P_k = s_k / Σ_k s_k  (sum-to-1 per voxel)

    Returns: (105, D, H, W) float32 in nnUNet internal space
    """
    D, H, W = validate_probs(probs_by_model, class_maps)
    global_prob = np.zeros((105, D, H, W), dtype=np.float32)

    # Background: product in log-space
    log_bg = np.zeros((D, H, W), dtype=np.float64)
    for probs in probs_by_model.values():
        log_bg += np.log(np.clip(probs[0].astype(np.float64), eps, 1.0))
    global_prob[0] = np.exp(log_bg).astype(np.float32)

    # Foreground: local → global
    for task_id, probs in probs_by_model.items():
        for local_c, global_c in class_maps[task_id].items():
            global_prob[global_c] = probs[local_c]

    # Normalize
    Z = global_prob.sum(axis=0, keepdims=True)
    global_prob /= np.maximum(Z, eps)
    return global_prob


def prob_to_seg(global_prob: np.ndarray) -> np.ndarray:
    """Argmax → uint16 label map (D, H, W) in internal space."""
    return global_prob.argmax(axis=0).astype(np.uint16)


# ─────────────────────────────────────────────────────────────────────────────
# Method A: fuse pre-averaged softmax per model
# ─────────────────────────────────────────────────────────────────────────────

def method_A(case_id: str,
             ensemble_root: Path,
             task_ids: List[int],
             class_maps: ModelClassMaps,
             eps: float = 1e-8) -> np.ndarray:
    """Load fold-averaged softmax, fuse → global prob (internal space).
    Reads: <ensemble_root>/Task{tid}/{case_id}.npz
    """
    probs_by_model: Dict[int, np.ndarray] = {}
    for tid in task_ids:
        probs_by_model[tid] = load_probs(
            ensemble_root / f"Task{tid}" / f"{case_id}.npz")
    return fuse_models_disjoint(probs_by_model, class_maps, eps)


# ─────────────────────────────────────────────────────────────────────────────
# Method B: fuse per-fold, then average
# ─────────────────────────────────────────────────────────────────────────────

def method_B(case_id: str,
             per_fold_root: Path,
             task_ids: List[int],
             class_maps: ModelClassMaps,
             folds: List[int],
             eps: float = 1e-8) -> np.ndarray:
    """Fuse 4 models per fold → average global probs → renormalize.
    Reads: <per_fold_root>/Task{tid}/fold_{f}/{case_id}.npz
    Note: differs slightly from A at boundaries due to nonlinearity of bg product.
    """
    fold_globals: List[np.ndarray] = []
    for f in folds:
        probs_by_model: Dict[int, np.ndarray] = {}
        for tid in task_ids:
            probs_by_model[tid] = load_probs(
                per_fold_root / f"Task{tid}" / f"fold_{f}" / f"{case_id}.npz")
        fold_globals.append(fuse_models_disjoint(probs_by_model, class_maps, eps))
    mean_prob = np.mean(fold_globals, axis=0).astype(np.float32)
    Z = mean_prob.sum(axis=0, keepdims=True)
    mean_prob /= np.maximum(Z, eps)
    return mean_prob


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(global_prob: np.ndarray, tag: str = "") -> None:
    prefix = f"[{tag}] " if tag else ""
    max_dev = float(np.abs(global_prob.sum(axis=0) - 1.0).max())
    print(f"  {prefix}prob range:    [{global_prob.min():.6f}, {global_prob.max():.6f}]")
    print(f"  {prefix}max |Σprob−1|: {max_dev:.2e}")


def compare_methods(case_id: str,
                    ensemble_root: Path,
                    per_fold_root: Path,
                    task_ids: List[int],
                    class_maps: ModelClassMaps,
                    folds: List[int],
                    eps: float = 1e-8,
                    print_hist: bool = False) -> None:
    print(f"\n=== compare: {case_id} ===")
    prob_A = method_A(case_id, ensemble_root, task_ids, class_maps, eps)
    prob_B = method_B(case_id, per_fold_root, task_ids, class_maps, folds, eps)
    print("  Method A:"); sanity_check(prob_A, "A")
    print("  Method B:"); sanity_check(prob_B, "B")
    diff = np.abs(prob_A - prob_B)
    print(f"  Max  |A−B|: {diff.max():.6f}  Mean |A−B|: {diff.mean():.8f}")
    seg_A, seg_B = prob_to_seg(prob_A), prob_to_seg(prob_B)
    agree = seg_A == seg_B
    print(f"  Identical argmax: {agree.mean()*100:.4f}%  "
          f"({(~agree).sum():,}/{agree.size:,} voxels differ)")
    if print_hist and not agree.all():
        for tag, vals in [("A", seg_A[~agree]), ("B", seg_B[~agree])]:
            us, cs = np.unique(vals, return_counts=True)
            print(f"\n  Top-10 labels at disagreement (Method {tag}):")
            for v, c in sorted(zip(us, cs), key=lambda x: -x[1])[:10]:
                print(f"    class {v:>4d}: {c:>10,}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cases_list",    type=Path, required=True,
                   help="Text file: one case ID per line")
    p.add_argument("--input_root",    type=Path, required=True,
                   help="Method A: <root>/Task{tid}/{case}.npz. "
                        "Method B: <root>/Task{tid}/fold_{f}/{case}.npz.")
    p.add_argument("--per_fold_root", type=Path, default=None,
                   help="Per-fold softmax root for B/compare. Defaults to --input_root.")
    p.add_argument("--ref_nii_root",  type=Path, default=None,
                   help="Root with per-fold hard-label .nii.gz (patient space). "
                        "Structure: <root>/Task{tid}/fold_{f}/{case}.nii.gz. "
                        "Required when --save_nii is set.")
    p.add_argument("--output_root",   type=Path, required=True)
    p.add_argument("--class_map_json",  type=Path, default=None)
    p.add_argument("--nnunet_raw_data", type=Path, default=None)
    p.add_argument("--save_class_map",  type=Path, default=None)
    p.add_argument("--method",  choices=["A", "B", "compare"], default="A")
    p.add_argument("--task_ids", nargs="+", type=int, default=[611, 612, 613, 614])
    p.add_argument("--folds",    nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--eps",      type=float, default=1e-8)
    p.add_argument("--save_prob", action="store_true",
                   help="Save fused probability as {case}_prob.npz (internal space).")
    p.add_argument("--save_nii",  action="store_true",
                   help="Save label map as {case}.nii.gz in patient space "
                        "(requires nibabel + --ref_nii_root).")
    p.add_argument("--sanity",     action="store_true")
    p.add_argument("--print_hist", action="store_true")
    return p.parse_args()


def run_case(case_id: str, args: argparse.Namespace,
             class_maps: ModelClassMaps, per_fold_root: Path) -> None:

    # ── 1. Fuse softmax in internal space ───────────────────────────────────
    global_prob = (
        method_A(case_id, args.input_root, args.task_ids, class_maps, args.eps)
        if args.method == "A" else
        method_B(case_id, per_fold_root, args.task_ids, class_maps,
                 args.folds, args.eps)
    )

    if args.sanity:
        sanity_check(global_prob, args.method)

    # ── 2. Argmax in internal space (D, H, W) ───────────────────────────────
    seg_internal = prob_to_seg(global_prob)   # (D, H, W) = (Z, Y, X)

    uniq, cnts = np.unique(seg_internal, return_counts=True)
    fg_cls = int((uniq > 0).sum())
    fg_vox = int(cnts[uniq > 0].sum()) if fg_cls else 0
    print(f"  internal shape: {seg_internal.shape}  "
          f"fg_classes: {fg_cls}  fg_voxels: {fg_vox:,}", end="")

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.save_prob:
        prob_path = args.output_root / f"{case_id}_prob.npz"
        np.savez_compressed(str(prob_path), prob=global_prob)
        print(f"\n  Prob  → {prob_path}", end="")

    if args.save_nii:
        if args.ref_nii_root is None:
            raise ValueError("--ref_nii_root is required when --save_nii is set")

        # ── 3. Resample internal (D, H, W) → patient (X, Y, Z) ─────────────
        ref_path = find_ref_nii(case_id, args.ref_nii_root,
                                args.task_ids, args.folds)
        seg_patient_xyz, affine = resample_seg_to_patient_space(
            seg_internal, ref_path)

        nii_path = args.output_root / f"{case_id}.nii.gz"
        save_nii(seg_patient_xyz, affine, nii_path)
        print(f"\n  NIfTI → {nii_path}  patient shape: {seg_patient_xyz.shape}", end="")
    else:
        npy_path = args.output_root / f"{case_id}.npy"
        np.save(str(npy_path), seg_internal)
        print(f"\n  Label → {npy_path}", end="")

    print()


def main() -> None:
    args = parse_args()

    # Class maps
    if args.class_map_json:
        class_maps = load_class_map_json(args.class_map_json)
        print(f"Class map: {args.class_map_json}")
    elif args.nnunet_raw_data:
        class_maps = load_class_map_from_nnunet(args.nnunet_raw_data, args.task_ids)
        print(f"Class map built from: {args.nnunet_raw_data}")
    else:
        print("ERROR: --class_map_json or --nnunet_raw_data required", file=sys.stderr)
        sys.exit(1)

    if args.save_class_map:
        serializable = {str(tid): {str(l): g for l, g in cm.items()}
                        for tid, cm in class_maps.items()}
        args.save_class_map.write_text(json.dumps(serializable, indent=2))
        print(f"Class map saved: {args.save_class_map}")

    total_fg = sum(len(cm) for cm in class_maps.values())
    print(f"Tasks: {args.task_ids}  |  fg classes: {total_fg}")
    for tid, cm in class_maps.items():
        print(f"  Task{tid}: {len(cm)} fg, global {min(cm.values())}..{max(cm.values())}")

    per_fold_root = args.per_fold_root or args.input_root
    cases = [ln.strip() for ln in args.cases_list.read_text().splitlines() if ln.strip()]
    print(f"\nCases: {len(cases)}  |  Method: {args.method}\n")

    args.output_root.mkdir(parents=True, exist_ok=True)
    errors: List[Tuple[str, str]] = []

    for i, case_id in enumerate(cases):
        print(f"[{i+1:>4}/{len(cases)}] {case_id}", end="  ", flush=True)
        try:
            if args.method == "compare":
                compare_methods(case_id, args.input_root, per_fold_root,
                                args.task_ids, class_maps, args.folds,
                                args.eps, args.print_hist)
            else:
                run_case(case_id, args, class_maps, per_fold_root)
        except Exception as exc:
            print(f"\n  ERROR: {exc}", file=sys.stderr)
            errors.append((case_id, str(exc)))

    print(f"\nDone. Output: {args.output_root}")
    if errors:
        print(f"\n{len(errors)} case(s) failed:")
        for cid, msg in errors:
            print(f"  {cid}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

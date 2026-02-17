"""
Create a mini nnUNet dataset (Task602_TotalSegMini) by symlinking to
existing converted NIfTI files from Dataset601_TotalSegmentatorV1.

5 training cases + 1 test case for quick iteration and debugging.

Usage:
    python scripts/setup_mini_dataset.py
"""

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_BASE = Path("/m/triton/scratch/elec/t41026-hintlab/tianmid1/data")
SOURCE_DIR = DATA_BASE / "nnUNet_raw" / "Dataset601_TotalSegmentatorV1"
OUTPUT_DIR = DATA_BASE / "nnUNet_raw" / "Dataset602_TotalSegMini"
TASK_LINK = DATA_BASE / "nnUNet_raw_data" / "Task602_TotalSegMini"

# First 5 train/val cases and first test case from meta.csv
TRAIN_IDS = ["0000", "0001", "0002", "0003", "0004"]
TEST_IDS = ["0013"]

# ---------------------------------------------------------------------------
# Label map (same 104 classes as Task601)
# ---------------------------------------------------------------------------

LABELS = {
    "0": "background",
    "1": "spleen", "2": "kidney_right", "3": "kidney_left",
    "4": "gallbladder", "5": "liver", "6": "stomach",
    "7": "aorta", "8": "inferior_vena_cava",
    "9": "portal_vein_and_splenic_vein", "10": "pancreas",
    "11": "adrenal_gland_right", "12": "adrenal_gland_left",
    "13": "lung_upper_lobe_left", "14": "lung_lower_lobe_left",
    "15": "lung_upper_lobe_right", "16": "lung_middle_lobe_right",
    "17": "lung_lower_lobe_right",
    "18": "vertebrae_L5", "19": "vertebrae_L4", "20": "vertebrae_L3",
    "21": "vertebrae_L2", "22": "vertebrae_L1",
    "23": "vertebrae_T12", "24": "vertebrae_T11", "25": "vertebrae_T10",
    "26": "vertebrae_T9", "27": "vertebrae_T8", "28": "vertebrae_T7",
    "29": "vertebrae_T6", "30": "vertebrae_T5", "31": "vertebrae_T4",
    "32": "vertebrae_T3", "33": "vertebrae_T2", "34": "vertebrae_T1",
    "35": "vertebrae_C7", "36": "vertebrae_C6", "37": "vertebrae_C5",
    "38": "vertebrae_C4", "39": "vertebrae_C3", "40": "vertebrae_C2",
    "41": "vertebrae_C1",
    "42": "esophagus", "43": "trachea",
    "44": "heart_myocardium", "45": "heart_atrium_left",
    "46": "heart_ventricle_left", "47": "heart_atrium_right",
    "48": "heart_ventricle_right", "49": "pulmonary_artery",
    "50": "brain",
    "51": "iliac_artery_left", "52": "iliac_artery_right",
    "53": "iliac_vena_left", "54": "iliac_vena_right",
    "55": "small_bowel", "56": "duodenum", "57": "colon",
    "58": "rib_left_1", "59": "rib_left_2", "60": "rib_left_3",
    "61": "rib_left_4", "62": "rib_left_5", "63": "rib_left_6",
    "64": "rib_left_7", "65": "rib_left_8", "66": "rib_left_9",
    "67": "rib_left_10", "68": "rib_left_11", "69": "rib_left_12",
    "70": "rib_right_1", "71": "rib_right_2", "72": "rib_right_3",
    "73": "rib_right_4", "74": "rib_right_5", "75": "rib_right_6",
    "76": "rib_right_7", "77": "rib_right_8", "78": "rib_right_9",
    "79": "rib_right_10", "80": "rib_right_11", "81": "rib_right_12",
    "82": "humerus_left", "83": "humerus_right",
    "84": "scapula_left", "85": "scapula_right",
    "86": "clavicula_left", "87": "clavicula_right",
    "88": "femur_left", "89": "femur_right",
    "90": "hip_left", "91": "hip_right", "92": "sacrum", "93": "face",
    "94": "gluteus_maximus_left", "95": "gluteus_maximus_right",
    "96": "gluteus_medius_left", "97": "gluteus_medius_right",
    "98": "gluteus_minimus_left", "99": "gluteus_minimus_right",
    "100": "autochthon_left", "101": "autochthon_right",
    "102": "iliopsoas_left", "103": "iliopsoas_right",
    "104": "urinary_bladder",
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Create output directories
    for subdir in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Symlink training files
    for case_id in TRAIN_IDS:
        img_name = f"TotalSegmentator_{case_id}_0000.nii.gz"
        lbl_name = f"TotalSegmentator_{case_id}.nii.gz"

        img_src = SOURCE_DIR / "imagesTr" / img_name
        lbl_src = SOURCE_DIR / "labelsTr" / lbl_name
        img_dst = OUTPUT_DIR / "imagesTr" / img_name
        lbl_dst = OUTPUT_DIR / "labelsTr" / lbl_name

        if not img_src.exists():
            raise FileNotFoundError(f"Source image not found: {img_src}")
        if not lbl_src.exists():
            raise FileNotFoundError(f"Source label not found: {lbl_src}")

        img_dst.unlink(missing_ok=True)
        lbl_dst.unlink(missing_ok=True)
        img_dst.symlink_to(img_src)
        lbl_dst.symlink_to(lbl_src)
        print(f"  Train: {img_name}")

    # Symlink test files
    for case_id in TEST_IDS:
        img_name = f"TotalSegmentator_{case_id}_0000.nii.gz"
        lbl_name = f"TotalSegmentator_{case_id}.nii.gz"

        img_src = SOURCE_DIR / "imagesTs" / img_name
        lbl_src = SOURCE_DIR / "labelsTs" / lbl_name
        img_dst = OUTPUT_DIR / "imagesTs" / img_name
        lbl_dst = OUTPUT_DIR / "labelsTs" / lbl_name

        if not img_src.exists():
            raise FileNotFoundError(f"Source image not found: {img_src}")
        if not lbl_src.exists():
            raise FileNotFoundError(f"Source label not found: {lbl_src}")

        img_dst.unlink(missing_ok=True)
        lbl_dst.unlink(missing_ok=True)
        img_dst.symlink_to(img_src)
        lbl_dst.symlink_to(lbl_src)
        print(f"  Test:  {img_name}")

    # Write dataset.json (nnUNet v1 format)
    training = [
        {
            "image": f"./imagesTr/TotalSegmentator_{cid}.nii.gz",
            "label": f"./labelsTr/TotalSegmentator_{cid}.nii.gz",
        }
        for cid in TRAIN_IDS
    ]
    test = [
        f"./imagesTs/TotalSegmentator_{cid}.nii.gz"
        for cid in TEST_IDS
    ]

    dataset_json = {
        "name": "Task602_TotalSegMini",
        "description": "Mini TotalSegmentator (5 train + 1 test) for debugging",
        "tensorImageSize": "4D",
        "reference": "",
        "licence": "",
        "release": "",
        "modality": {"0": "CT"},
        "labels": LABELS,
        "numTraining": len(TRAIN_IDS),
        "numTest": len(TEST_IDS),
        "training": training,
        "test": test,
    }

    json_path = OUTPUT_DIR / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"\nWrote {json_path}")

    # Create nnUNet_raw_data symlink (nnUNet v1 expects Task* under nnUNet_raw_data/)
    TASK_LINK.parent.mkdir(parents=True, exist_ok=True)
    if TASK_LINK.is_symlink() or TASK_LINK.exists():
        TASK_LINK.unlink()
    TASK_LINK.symlink_to(OUTPUT_DIR)
    print(f"Symlink: {TASK_LINK} -> {OUTPUT_DIR}")

    print(f"\nMini dataset ready: {OUTPUT_DIR}")
    print(f"  Train: {len(TRAIN_IDS)} cases")
    print(f"  Test:  {len(TEST_IDS)} cases")
    print("\nNext steps:")
    print("  sbatch scripts/01_preprocess_mini.slurm")
    print("  sbatch --export=ALL,FOLD=0 scripts/02_train_mini.slurm")


if __name__ == "__main__":
    main()

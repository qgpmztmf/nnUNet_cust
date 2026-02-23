#!/usr/bin/env python3
"""
apply_tuning_decision.py
========================
Reads a tuning-decision JSON produced by the Auto-Tuning Supervisor Agent and
applies its recommendations to hyperparameter_reference.json by writing
``active_value`` entries for every parameter listed in ``tuning_decision``.

Before writing the updated reference file the existing
``hyperparameter_reference.json`` is archived as
``hyperparameter_reference_round_<N>.json``, where N is the next available
integer in the documentation directory.

Usage
-----
    uv run python agent/apply_tuning_decision.py <tuning_decision.json> [--doc-dir <dir>]

    # Example
    uv run python agent/apply_tuning_decision.py \\
        documentation/tuning_decision_task601_round1.json

Arguments
---------
tuning_decision   Path to the tuning-decision JSON (required).
--doc-dir         Directory that contains hyperparameter_reference.json.
                  Defaults to  <script_dir>/../documentation/
--dry-run         Print the diff without writing any files.
"""

import argparse
import json
import os
import re
import shutil
import sys


# ── helpers ──────────────────────────────────────────────────────────────────

def _doc_dir_default() -> str:
    """Return the documentation/ directory relative to this script."""
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "documentation")
    )


def _next_round_number(doc_dir: str) -> int:
    """Return the next unused round index for archiving the reference file."""
    pattern = re.compile(r"^hyperparameter_reference_round_(\d+)\.json$")
    existing = []
    for name in os.listdir(doc_dir):
        m = pattern.match(name)
        if m:
            existing.append(int(m.group(1)))
    return (max(existing) + 1) if existing else 1


def _load_json(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def _save_json(path: str, data: dict) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


# ── main logic ────────────────────────────────────────────────────────────────

def apply(tuning_path: str, doc_dir: str, dry_run: bool = False) -> None:
    ref_path = os.path.join(doc_dir, "hyperparameter_reference.json")

    # ── validate inputs ───────────────────────────────────────────────────────
    if not os.path.isfile(tuning_path):
        sys.exit(f"ERROR: tuning decision file not found: {tuning_path}")
    if not os.path.isfile(ref_path):
        sys.exit(f"ERROR: hyperparameter reference file not found: {ref_path}")

    tuning = _load_json(tuning_path)
    ref    = _load_json(ref_path)

    decisions: dict = tuning.get("tuning_decision", {})
    if not decisions:
        sys.exit("ERROR: 'tuning_decision' key missing or empty in the tuning file.")

    # ── compute diff ──────────────────────────────────────────────────────────
    changes = []     # list of (param_name, new_value, is_new_param)
    unknown = []     # params in decision but not in reference

    for param, info in decisions.items():
        if "new_value" not in info:
            print(f"  SKIP  {param!r}: no 'new_value' field in tuning decision")
            continue
        new_val = info["new_value"]
        if param in ref:
            old_active = ref[param].get("active_value", "<not set>")
            changes.append((param, new_val, old_active, False))
        else:
            changes.append((param, new_val, "<not set>", True))
            unknown.append(param)

    if not changes:
        print("Nothing to apply — all tuning_decision entries are missing 'new_value'.")
        return

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\nTuning decision:  {os.path.abspath(tuning_path)}")
    print(f"Reference file:   {ref_path}")
    print(f"\n{'PARAM':<40}  {'OLD active_value':<20}  {'NEW active_value'}")
    print("-" * 80)
    for param, new_val, old_active, is_new in changes:
        flag = " [NEW PARAM]" if is_new else ""
        print(f"  {param:<38}  {str(old_active):<20}  {new_val}{flag}")

    if unknown:
        print(f"\n  Note: {len(unknown)} param(s) not found in reference — "
              "they will be added as new entries with only 'active_value' set.")

    if dry_run:
        print("\n[dry-run] No files written.")
        return

    # ── archive the current reference file ───────────────────────────────────
    round_n   = _next_round_number(doc_dir)
    archive   = os.path.join(doc_dir, f"hyperparameter_reference_round_{round_n}.json")
    shutil.copy2(ref_path, archive)
    print(f"\nArchived current reference → {os.path.basename(archive)}")

    # ── apply active_value overrides ──────────────────────────────────────────
    for param, new_val, _old, is_new in changes:
        if is_new:
            ref[param] = {"active_value": new_val}
        else:
            ref[param]["active_value"] = new_val

    # ── write updated reference ───────────────────────────────────────────────
    _save_json(ref_path, ref)
    print(f"Updated reference → {ref_path}")
    print(f"\nDone. Applied {len(changes)} change(s) for round {round_n}.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply an Auto-Tuning Supervisor Agent decision to hyperparameter_reference.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tuning_decision",
        help="Path to the tuning-decision JSON file.",
    )
    parser.add_argument(
        "--doc-dir",
        default=_doc_dir_default(),
        help="Directory containing hyperparameter_reference.json "
             f"(default: {_doc_dir_default()})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned changes without modifying any files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    apply(
        tuning_path=args.tuning_decision,
        doc_dir=args.doc_dir,
        dry_run=args.dry_run,
    )

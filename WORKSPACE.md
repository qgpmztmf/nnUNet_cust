# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Overview

This is a research workspace on the Triton HPC cluster containing multiple independent ML/medical imaging projects. The **primary active project** is `nnUNet_cust/` — see its own `CLAUDE.md` for detailed guidance on that project.

```
/scratch/work/tianmid1/
├── nnUNet_cust/                    # PRIMARY — custom nnUNet v1 pipeline
├── mimic-iv-analysis/              # MIMIC-IV EHR data extraction & analysis
├── EHR-R1/                         # Reasoning-enhanced LLMs for EHR tasks
├── textgrad/                       # Text-based automatic differentiation
├── TotalSegmentator/               # Reference: original TotalSeg tool
├── TotalSegmentator-to-nnUNet-format-convert/  # Dataset format conversion
├── HINTools/                       # Health informatics tools
├── CS336/                          # Stanford CS336 practice code
└── data/                           # Shared datasets (symlinked from /m/triton/scratch/...)
```

Each subdirectory is its own git repository.

## Common Environment

- **Python:** 3.12 across all projects
- **Package manager:** `uv` (run scripts as `uv run <script>` from the project root)
- **Cluster:** Triton HPC, SLURM job scheduler
- **Data base:** `/m/triton/scratch/elec/t41026-hintlab/tianmid1/data/`

## nnUNet_cust (Primary Project)

> See `nnUNet_cust/CLAUDE.md` for full details. Key facts:

- End-to-end nnUNet **v1** pipeline for TotalSegmentator segmentation
- Always use `uv run` from `nnUNet_cust/` as the working directory
- Use `nibabel` for NIfTI I/O — **never SimpleITK** (fails on many CT scans due to orthonormality checks)
- Training jobs submitted via SLURM scripts in `nnUNet_cust/scripts/` (numbered 00→04)

**Common commands:**
```bash
# Preprocess a task
sbatch --array=0-4 --export=ALL,TASK=601 scripts/01_preprocess.slurm

# Train all grouped tasks across all folds
for T in 611 612 613 614; do
    sbatch --array=0-4 --export=ALL,TASK=$T scripts/02_train_grouped.slurm
done

# Validate syntax before submitting
bash -n scripts/<script>.slurm
sbatch --test-only scripts/<script>.slurm
```

**Active tasks:** Task601 (104-class full), Task611–614 (grouped subsets of TotalSeg labels)

## mimic-iv-analysis

EHR data extraction from MIMIC-IV. Entry points:
- `main.py` — primary entry point
- `batch_extract_subjects.py` — batch processing via `slurm_split.sh`

```bash
uv run python main.py
```

## EHR-R1

LLM fine-tuning (SFT + GRPO) for EHR reasoning tasks. Uses HuggingFace Transformers + Accelerate + DeepSpeed.

```bash
# Supervised fine-tuning
accelerate launch --config_file=./scripts/accelerate_configs/deepspeed_zero3.yaml \
  sft.py --bf16 True --max_seq_length 8192 ...

# Evaluation
python test.py --dataset_name ... --model_name_or_path ... --use_vllm
```

Three training stages: SFT on general EHR → SFT on reasoning chains → GRPO reinforcement.

## Data Paths

```bash
DATA_BASE=/m/triton/scratch/elec/t41026-hintlab/tianmid1/data
# Also accessible at /scratch/work/tianmid1/data (symlinked)

${DATA_BASE}/nnUNet_raw_data/        # nnUNet raw input
${DATA_BASE}/nnUNet_preprocessed/    # nnUNet preprocessed
${DATA_BASE}/nnUNet_results/         # Training checkpoints & results
${DATA_BASE}/TotalSegmentator_v1/    # Source dataset (1204 subjects)
${DATA_BASE}/physionet.org/          # MIMIC-IV EHR data
```

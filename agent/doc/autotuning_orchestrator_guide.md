# `autotuning_orchestrator.py` — User Guide

## What it does

A **multi-agent pipeline** that uses an LLM (Claude or DeepSeek) to automatically analyze
nnUNet training results and propose, validate, and apply hyperparameter changes — then
generates ready-to-submit training and evaluation SLURM scripts for the next training round.

---

## Prerequisites

**1. API key** — set in `nnUNet_cust/.env` or as an environment variable:

```bash
# For Claude (default):
export ANTHROPIC_API_KEY=sk-ant-...

# OR for DeepSeek:
export DEEPSEEK_API_KEY=sk-...
```

**2. `hyperparameter_reference.json`** — the config file the agents read and modify. Lives at:

```
agent/doc/hyperparameter_reference.json
```

Each parameter has an `active_value` field. If absent, the trainer uses the documented
default. Edit `active_value` fields manually before Round 1 to set your starting point.

---

## Two-phase workflow

| Phase | What happens |
|---|---|
| **Round 1** | No LLM agents. Generates training scripts directly from `hyperparameter_reference.json` as-is. Train, evaluate, then come back for Round 2. |
| **Round 2+** | Agents read the previous round's results → diagnose → propose changes → update `hyperparameter_reference.json` → generate new training scripts. |

---

## Step-by-step usage

### Phase 1 — Generate the first training script

```bash
cd /home/tianmid1/tianmid1/t41026-hintlab/tianmid1/nnUNet_cust

# Generate bash + SLURM scripts (no training yet):
uv run python agent/autotuning_orchestrator.py \
    --task Task601_TotalSegmentatorV1 \
    --slurm

# Outputs go to:  agent/runs/round_1_<timestamp>/
#   train_task601_round1.sh        ← bash script
#   train_task601_round1.slurm     ← SLURM array script
#   eval_task601_round1.slurm      ← post-training evaluation script
```

Submit training (all 5 folds):

```bash
sbatch --array=0-4 agent/runs/round_1_<timestamp>/train_task601_round1.slurm
```

After all folds finish, run evaluation:

```bash
sbatch agent/runs/round_1_<timestamp>/eval_task601_round1.slurm
```

The eval script copies `val_summary.json` (and optionally `test_summary.json`) into the
round directory so Round 2 can find them automatically.

### Phase 2 — Agent-guided tuning (Round 2+)

```bash
uv run python agent/autotuning_orchestrator.py \
    --task Task601_TotalSegmentatorV1 \
    --slurm
```

The orchestrator auto-detects the round number by scanning `agent/runs/`. It:

1. Reads the previous round's `val_summary.json` / `test_summary.json`
2. **Step 1** — `hp_autotuning` agent: diagnoses weak classes, proposes new `active_value`s
3. **Step 2** — `hp_checker` agent: validates proposed values are in-range
4. **Step 3** — Applies changes to `hyperparameter_reference.json` (archives the old one as `hyperparameter_reference_round_N.json`)
5. **Step 4** — `training_launcher` agent: generates new training scripts
6. Outputs to `agent/runs/round_2_<timestamp>/`

---

## All CLI options

```bash
uv run python agent/autotuning_orchestrator.py \
    --task Task601_TotalSegmentatorV1 \    # nnUNet task name
    --slurm \                               # also generate .slurm array script
    --max-rounds 1 \                        # run N rounds in one call
    --provider anthropic \                  # or: deepseek
    --model claude-opus-4-6 \              # override model
    --auto-train \                          # submit sbatch automatically (careful!)
    --use-test-data \                       # include test-set inference in eval script
    --test-summary path/to/summary.json \   # override auto-detected test result
    --val-summary path/to/summary.json \    # override auto-detected val result
    --hp-ref agent/doc/hyperparameter_reference.json \
    --force-round 2                         # force a specific round number
```

| Option | Default | Description |
|---|---|---|
| `--task` | `Task601_TotalSegmentatorV1` | nnUNet task name |
| `--slurm` | off | Generate `.slurm` array script in addition to bash script |
| `--max-rounds` | `1` | Number of tuning rounds to run in one invocation |
| `--provider` | `anthropic` | LLM backend: `anthropic` or `deepseek` |
| `--model` | `claude-opus-4-6` | Model ID override |
| `--auto-train` | off | Automatically execute/submit the generated training script |
| `--use-test-data` | off | Include test-set inference (GPU) in the eval script |
| `--test-summary` | auto-detected | Path to test results `summary.json` |
| `--val-summary` | auto-detected | Path to cross-validation `summary.json` |
| `--hp-ref` | `agent/doc/hyperparameter_reference.json` | Path to hyperparameter reference |
| `--force-round` | auto | Force a specific round number (useful for reruns) |

---

## Output files per round

```
agent/runs/round_N_<timestamp>/
├── orchestrator.log                        # full run log
├── step1_hp_autotuning_input.txt           # what was sent to the agent
├── step1_hp_autotuning_response.txt        # raw LLM response
├── step2_hp_checker_input.txt
├── step2_hp_checker_response.txt
├── checked_hyperparameters.json            # validity flags per parameter
├── tuning_decision_task601_roundN.json     # diagnosis + proposed changes
├── train_task601_roundN.sh                 # bash training script
├── train_task601_roundN.slurm             # SLURM array job (if --slurm)
├── eval_task601_roundN.slurm              # evaluation script
├── val_summary.json                        # copy of val results (after eval)
└── test_summary.json                       # copy of test results (after eval)
```

---

## Typical multi-round workflow

```bash
# Round 1: generate scripts from current hyperparameter_reference.json
uv run python agent/autotuning_orchestrator.py --task Task601_TotalSegmentatorV1 --slurm
sbatch --array=0-4 agent/runs/round_1_*/train_task601_round1.slurm

# ... wait for training to complete ...

sbatch agent/runs/round_1_*/eval_task601_round1.slurm

# ... wait for eval (copies summaries into round_1 dir) ...

# Round 2: agent analyzes results and proposes new hyperparameters
uv run python agent/autotuning_orchestrator.py --task Task601_TotalSegmentatorV1 --slurm
sbatch --array=0-4 agent/runs/round_2_*/train_task601_round2.slurm

# ... repeat for subsequent rounds ...
```

---

## Notes

- The trainer used is `nnUNetTrainerV2_configurable`, which reads
  `hyperparameter_reference.json` via the `NNUNET_HP_REF` environment variable set
  automatically in the generated SLURM scripts.
- If `eval_task601_roundN.slurm` has not been run yet (or training is still in progress),
  you can manually copy `summary.json` files into `agent/runs/round_N_<timestamp>/` as
  `val_summary.json` / `test_summary.json` before invoking the next round.
- Previous `hyperparameter_reference.json` versions are archived as
  `agent/doc/hyperparameter_reference_round_N.json` before each update so changes are
  always recoverable.
- Use `--force-round N` to re-run or debug a specific round without affecting the
  auto-detected round counter.

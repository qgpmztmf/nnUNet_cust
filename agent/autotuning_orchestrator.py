#!/usr/bin/env python3
"""
autotuning_orchestrator.py
==========================
Multi-agent pipeline orchestrator for nnUNet hyperparameter auto-tuning.

Pipeline (per round):
  Step 1 — agent_hp-autotuning   → tuning_decision_taskXXX_roundN.json
  Step 2 — agent_hp-checker      → checked_hyperparameters.json
  Step 3 — agent_modify_hp       → updated hyperparameter_reference.json
  Step 4 — agent_training_launcher → train_taskXXX_roundN.sh
  Step 5 — (optional) execute training script

Usage:
  python autotuning_orchestrator.py [options]

Options:
  --task        nnUNet task name  (default: Task601_TotalSegmentatorV1)
  --auto-train  Execute generated training script automatically
  --slurm       Also generate a SLURM array script  (train_taskXXX_roundN.slurm)
  --max-rounds  Number of tuning rounds (default: 1)
  --provider    LLM provider: anthropic (default) or deepseek
  --model       Model ID (default depends on provider)
  --test-summary  Path to test results summary JSON
  --val-summary   Path to validation summary JSON
  --hp-ref        Path to hyperparameter_reference.json

All per-round outputs are written to:
  agent/runs/round_N_<timestamp>/
    orchestrator.log
    step1_hp_autotuning_{input,response}.txt
    step2_hp_checker_{input,response}.txt
    checked_hyperparameters.json
    step4_training_launcher_{input,response}.txt
    train_taskXXX_roundN.sh          (bash training script)
    train_taskXXX_roundN.slurm       (SLURM array script, if --slurm)

Environment variables:
  ANTHROPIC_API_KEY   Required when --provider anthropic
  DEEPSEEK_API_KEY    Required when --provider deepseek
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

try:
    from openai import OpenAI as _OpenAIClient
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ── Load .env from repo root (nnUNet_cust/.env) ───────────────────────────────
# Variables already set in the shell environment take precedence (override=False).
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE, override=False)

# ── Directory layout ──────────────────────────────────────────────────────────

AGENT_DIR  = Path(__file__).resolve().parent           # nnUNet_cust/agent/
REPO_ROOT  = AGENT_DIR.parent                          # nnUNet_cust/
DOC_DIR    = REPO_ROOT / "documentation"
LOG_DIR    = AGENT_DIR / "logs"
RUNS_DIR   = AGENT_DIR / "runs"
PROMPT_DIR = AGENT_DIR / "prompts"

PROMPTS: dict[str, Path] = {
    "hp_autotuning":    PROMPT_DIR / "agent_hp-autotuning.md",
    "hp_checker":       PROMPT_DIR / "agent-hp-checker.md",
    "modify_hp":        PROMPT_DIR / "agent-modify-hy.md",
    "training_launcher": PROMPT_DIR / "agent-training-launcher.md",
}

# ── API / model settings ──────────────────────────────────────────────────────

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_DEEPSEEK  = "deepseek"
SUPPORTED_PROVIDERS = (PROVIDER_ANTHROPIC, PROVIDER_DEEPSEEK)

DEFAULT_MODEL            = "claude-opus-4-6"       # used when provider=anthropic
DEFAULT_DEEPSEEK_MODEL   = "deepseek-chat"          # used when provider=deepseek
DEEPSEEK_BASE_URL        = "https://api.deepseek.com"

MAX_TOKENS      = 8192
TEMPERATURE     = 0        # deterministic
MAX_RETRIES     = 3
RETRY_BASE_DELAY = 10.0    # seconds (multiplied by attempt number)
REQUEST_TIMEOUT  = 300.0   # seconds

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    logfile = logging.FileHandler(run_dir / "orchestrator.log", encoding="utf-8")
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    return logger


# ── API wrappers ──────────────────────────────────────────────────────────────

def _load_prompt(prompt_path: Path) -> str:
    """Read prompt file and strip any surrounding quotes added by editors."""
    text = prompt_path.read_text(encoding="utf-8").strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    return text


def _call_anthropic(
    system_prompt: str,
    input_context: str,
    model: str,
    timeout: float,
    log: logging.Logger,
    attempt: int,
    max_retries: int,
    prompt_name: str,
) -> str:
    log.debug(
        "  [Anthropic] attempt %d/%d  model=%s  prompt=%s",
        attempt, max_retries, model, prompt_name,
    )
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        timeout=timeout,
    )
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": input_context}],
    )
    return response.content[0].text


def _call_deepseek(
    system_prompt: str,
    input_context: str,
    model: str,
    timeout: float,
    log: logging.Logger,
    attempt: int,
    max_retries: int,
    prompt_name: str,
) -> str:
    if not _OPENAI_AVAILABLE:
        raise ImportError(
            "openai package is required for DeepSeek provider. "
            "Install it with: pip install openai"
        )
    log.debug(
        "  [DeepSeek] attempt %d/%d  model=%s  prompt=%s",
        attempt, max_retries, model, prompt_name,
    )
    client = _OpenAIClient(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=DEEPSEEK_BASE_URL,
        timeout=timeout,
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": input_context},
        ],
        stream=False,
    )
    return response.choices[0].message.content


def call_agent(
    prompt_path: Path,
    input_context: str,
    model: str = DEFAULT_MODEL,
    provider: str = PROVIDER_ANTHROPIC,
    max_retries: int = MAX_RETRIES,
    retry_base_delay: float = RETRY_BASE_DELAY,
    timeout: float = REQUEST_TIMEOUT,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Load a system prompt from *prompt_path*, combine with *input_context* as
    the user message, call the selected LLM API (streaming OFF), and return
    the raw assistant text.

    *provider* selects the backend:
      - ``"anthropic"``  — Claude via Anthropic SDK  (env: ANTHROPIC_API_KEY)
      - ``"deepseek"``   — DeepSeek via OpenAI SDK   (env: DEEPSEEK_API_KEY)

    Retries up to *max_retries* times on transient errors with linear back-off.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Supported: {SUPPORTED_PROVIDERS}"
        )

    log = logger or logging.getLogger("orchestrator")
    system_prompt = _load_prompt(prompt_path)

    _TRANSIENT_ANTHROPIC = (
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
    )

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(1, max_retries + 1):
        try:
            if provider == PROVIDER_ANTHROPIC:
                text = _call_anthropic(
                    system_prompt, input_context, model,
                    timeout, log, attempt, max_retries, prompt_path.name,
                )
            else:
                text = _call_deepseek(
                    system_prompt, input_context, model,
                    timeout, log, attempt, max_retries, prompt_path.name,
                )
            log.debug("  [API] received %d chars", len(text))
            return text

        except _TRANSIENT_ANTHROPIC as exc:
            last_exc = exc
            log.warning("  [API] error attempt %d/%d: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                sleep = retry_base_delay * attempt
                log.debug("  [API] sleeping %.0fs before retry", sleep)
                time.sleep(sleep)
        except Exception as exc:
            # For non-Anthropic errors (e.g. openai errors), retry the same way
            last_exc = exc
            log.warning("  [API] error attempt %d/%d: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                sleep = retry_base_delay * attempt
                log.debug("  [API] sleeping %.0fs before retry", sleep)
                time.sleep(sleep)

    raise RuntimeError(f"call_agent: all {max_retries} retries exhausted") from last_exc


# ── AgentRunner ───────────────────────────────────────────────────────────────

class AgentRunner:
    """
    Loads an agent prompt, injects runtime context, calls the LLM API,
    and persists both the raw input and raw response to disk.
    """

    def __init__(
        self,
        name: str,
        prompt_path: Path,
        run_dir: Path,
        model: str = DEFAULT_MODEL,
        provider: str = PROVIDER_ANTHROPIC,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.name        = name
        self.prompt_path = prompt_path
        self.run_dir     = run_dir
        self.model       = model
        self.provider    = provider
        self.log         = logger or logging.getLogger("orchestrator")

    def run(self, input_context: str) -> str:
        """Execute the agent and return the raw LLM response text."""
        self.log.info("[%s] starting  (provider=%s  model=%s)", self.name, self.provider, self.model)

        (self.run_dir / f"{self.name}_input.txt").write_text(
            input_context, encoding="utf-8"
        )

        t0  = time.monotonic()
        raw = call_agent(
            prompt_path     = self.prompt_path,
            input_context   = input_context,
            model           = self.model,
            provider        = self.provider,
            logger          = self.log,
        )
        elapsed = time.monotonic() - t0

        (self.run_dir / f"{self.name}_response.txt").write_text(raw, encoding="utf-8")
        self.log.info("[%s] done in %.1fs  (%d chars)", self.name, elapsed, len(raw))
        return raw

    # ── output extraction helpers ─────────────────────────────────────────────

    @staticmethod
    def extract_json(raw: str) -> dict:
        """
        Extract the first valid JSON object from the raw LLM response.
        Handles responses wrapped in markdown code fences.
        """
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = re.sub(r"```", "", cleaned)

        # Attempt a direct parse on the whole cleaned string
        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            pass

        # Fall back to finding the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"No valid JSON found in LLM response (first 500 chars):\n{raw[:500]}"
        )

    @staticmethod
    def extract_bash_script(raw: str) -> str:
        """
        Extract bash script content from the LLM response.
        Strips markdown code fences when present.
        """
        match = re.search(r"```(?:bash|sh)?\s*\n(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()

        stripped = raw.strip()
        if stripped.startswith("#!"):
            return stripped

        return stripped


# ── PipelineManager ───────────────────────────────────────────────────────────

class PipelineManager:
    """
    Orchestrates the sequential multi-agent auto-tuning pipeline.

    Responsibilities:
    - sequential agent execution with dependency passing
    - round number tracking and auto-increment
    - file management (documentation/, logs/, runs/)
    - error reporting and safe shutdown
    """

    def __init__(
        self,
        task: str,
        doc_dir: Path,
        log_dir: Path,
        runs_dir: Path,
        model: str = DEFAULT_MODEL,
        provider: str = PROVIDER_ANTHROPIC,
        auto_train: bool = False,
        generate_slurm: bool = False,
    ) -> None:
        self.task          = task
        self.doc_dir       = doc_dir
        self.log_dir       = log_dir
        self.runs_dir      = runs_dir
        self.model         = model
        self.provider      = provider
        self.auto_train    = auto_train
        self.generate_slurm = generate_slurm

        self.round = self._detect_next_round()

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = runs_dir / f"round_{self.round}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log = _setup_logging(self.run_dir)
        self.log.info(
            "=== Orchestrator start | task=%s | round=%d | provider=%s | model=%s ===",
            self.task, self.round, self.provider, self.model,
        )

    # ── round detection ───────────────────────────────────────────────────────

    def _detect_next_round(self) -> int:
        """Return the next unused round number by scanning existing artefacts."""
        indices: list[int] = []

        run_pat = re.compile(r"^round_(\d+)")
        if self.runs_dir.exists():
            for entry in self.runs_dir.iterdir():
                m = run_pat.match(entry.name)
                if m:
                    indices.append(int(m.group(1)))

        arc_pat = re.compile(r"^hyperparameter_reference_round_(\d+)\.json$")
        if self.doc_dir.exists():
            for name in os.listdir(self.doc_dir):
                m = arc_pat.match(name)
                if m:
                    indices.append(int(m.group(1)))

        return max(indices, default=0) + 1

    # ── helpers ───────────────────────────────────────────────────────────────

    def _task_id(self) -> str:
        """Extract numeric task ID string, e.g. '601' from 'Task601_...'."""
        m = re.search(r"Task(\d+)", self.task)
        return m.group(1) if m else self.task

    def _load_file(self, path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return path.read_text(encoding="utf-8")

    def _save_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self.log.info("Saved → %s", path)

    def _validate_json_keys(
        self, data: dict, required: list[str], step: str
    ) -> None:
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"[{step}] JSON output missing required keys: {missing}")

    def _agent(self, name: str, prompt_key: str) -> AgentRunner:
        return AgentRunner(
            name        = name,
            prompt_path = PROMPTS[prompt_key],
            run_dir     = self.run_dir,
            model       = self.model,
            provider    = self.provider,
            logger      = self.log,
        )

    def _find_previous_checker_result(self) -> Optional[Path]:
        """
        Look for a checked_hyperparameters.json from a previous run so that
        Step 1 can use it as input without needing Step 2 to run first.
        """
        if not self.runs_dir.exists():
            return None
        run_pat = re.compile(r"^round_(\d+)")
        candidates: list[tuple[int, Path]] = []
        for entry in sorted(self.runs_dir.iterdir()):
            m = run_pat.match(entry.name)
            if m:
                candidate = entry / "checked_hyperparameters.json"
                if candidate.exists():
                    candidates.append((int(m.group(1)), candidate))
        if not candidates:
            return None
        # Return the one with the highest round index
        candidates.sort(key=lambda t: t[0])
        return candidates[-1][1]

    # ── pipeline steps ────────────────────────────────────────────────────────

    def step1_hp_autotuning(
        self,
        test_summary_path: Path,
        val_summary_path: Path,
        hp_ref_path: Path,
        checker_result_path: Optional[Path] = None,
    ) -> tuple[dict, Path]:
        """
        Run agent_hp-autotuning.
        Returns (tuning_decision_dict, saved_path).
        """
        self.log.info("── STEP 1: HP Auto-Tuning ──────────────────────────────")

        hp_ref_content = self._load_file(hp_ref_path)

        sections: list[str] = [
            f"TASK: {self.task}",
            f"ROUND: {self.round}",
            "",
        ]

        sections.append(f"## TEST RESULT (path: {test_summary_path})")
        if test_summary_path.exists():
            sections.append(self._load_file(test_summary_path))
        else:
            self.log.warning("Test summary not found: %s", test_summary_path)
            sections.append("(file not found — no test results available for this run)")

        sections.append("")
        sections.append(f"## VALIDATION RESULT (path: {val_summary_path})")
        if val_summary_path.exists():
            sections.append(self._load_file(val_summary_path))
        else:
            self.log.warning("Validation summary not found: %s", val_summary_path)
            sections.append(
                "(file not found — no validation results available for this run)"
            )

        if checker_result_path and checker_result_path.exists():
            sections.append("")
            sections.append("## HYPERPARAMETER CHECKER RESULT (from previous round)")
            sections.append(self._load_file(checker_result_path))
        else:
            sections.append("")
            sections.append(
                "## HYPERPARAMETER CHECKER RESULT\n"
                "(not available — treat all parameters as tunable)"
            )

        sections += [
            "",
            "## CURRENT HYPERPARAMETER REFERENCE",
            hp_ref_content,
            "",
            "Output ONLY a valid JSON object following the required format.",
        ]

        context = "\n".join(sections)
        raw     = self._agent("step1_hp_autotuning", "hp_autotuning").run(context)
        decision = AgentRunner.extract_json(raw)
        self._validate_json_keys(
            decision, ["diagnosis", "tuning_decision", "training_strategy"], "step1"
        )

        out_path = (
            self.doc_dir
            / f"tuning_decision_task{self._task_id()}_round{self.round}.json"
        )
        self._save_json(out_path, decision)
        self.log.info("STEP 1 complete → %s", out_path)
        return decision, out_path

    def step2_hp_checker(self, hp_ref_path: Path) -> tuple[dict, Path]:
        """
        Run agent_hp-checker.
        Returns (checked_dict, saved_path).
        checked_dict maps parameter names to boolean validity flags.
        """
        self.log.info("── STEP 2: HP Checker ──────────────────────────────────")

        hp_ref_content = self._load_file(hp_ref_path)

        context = "\n".join([
            f"TASK: {self.task}",
            f"ROUND: {self.round}",
            "",
            "## HYPERPARAMETER REFERENCE (to verify)",
            hp_ref_content,
            "",
            "Verify every parameter listed above.",
            'Output ONLY a valid JSON object: { "param_name": true/false, ... }',
        ])

        raw     = self._agent("step2_hp_checker", "hp_checker").run(context)
        checked = AgentRunner.extract_json(raw)

        if not isinstance(checked, dict):
            raise ValueError("step2: checker output is not a JSON object")
        for key, val in checked.items():
            if not isinstance(val, bool):
                raise ValueError(
                    f"step2: expected boolean for key {key!r}, got {val!r}"
                )

        out_path = self.run_dir / "checked_hyperparameters.json"
        self._save_json(out_path, checked)
        self.log.info("STEP 2 complete → %s", out_path)
        return checked, out_path

    def step3_modify_hp(
        self,
        tuning_decision: dict,
        tuning_decision_path: Path,
        hp_ref_path: Path,
    ) -> Path:
        """
        Apply tuning decisions to hyperparameter_reference.json.

        Primary path: call the deterministic apply_tuning_decision.py script
        (archives the old reference and writes active_value entries).

        Fallback: call the LLM agent if the script is unavailable.

        Returns the path to the (now updated) hyperparameter_reference.json.
        """
        self.log.info("── STEP 3: Modify Hyperparameters ──────────────────────")

        apply_script = AGENT_DIR / "apply_tuning_decision.py"

        if apply_script.exists():
            self.log.info("Using apply_tuning_decision.py (deterministic)")
            cmd = [
                sys.executable, str(apply_script),
                str(tuning_decision_path),
                "--doc-dir", str(self.doc_dir),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                for line in result.stdout.splitlines():
                    self.log.info("  [apply_tuning] %s", line)
            if result.returncode != 0:
                self.log.error("apply_tuning_decision.py stderr:\n%s", result.stderr)
                raise RuntimeError(
                    f"step3: apply_tuning_decision.py exited with code {result.returncode}"
                )
        else:
            # LLM fallback: ask the modify_hp agent to emit an updated JSON
            self.log.info(
                "apply_tuning_decision.py not found — using LLM agent as fallback"
            )
            hp_ref_content = self._load_file(hp_ref_path)
            context = "\n".join([
                f"TASK: {self.task}",
                f"ROUND: {self.round}",
                "",
                "## TUNING DECISION",
                json.dumps(tuning_decision, indent=2),
                "",
                "## CURRENT HYPERPARAMETER REFERENCE",
                hp_ref_content,
                "",
                "Apply the tuning decisions: for each parameter in tuning_decision, "
                "set its active_value to the new_value.",
                "Output ONLY the complete updated hyperparameter_reference.json as "
                "valid JSON.",
            ])
            raw     = self._agent("step3_modify_hp", "modify_hp").run(context)
            updated = AgentRunner.extract_json(raw)
            self._save_json(hp_ref_path, updated)

        self.log.info("STEP 3 complete → %s", hp_ref_path)
        return hp_ref_path

    def step4_training_launcher(self, hp_ref_path: Path) -> Path:
        """
        Run agent_training_launcher.
        Returns the path to the generated bash training script.
        """
        self.log.info("── STEP 4: Training Launcher ────────────────────────────")

        hp_ref_content = self._load_file(hp_ref_path)
        task_id        = self._task_id()

        context = "\n".join([
            f"TASK: {self.task}",
            f"ROUND: {self.round}",
            f"TASK_ID: {task_id}",
            "",
            "## UPDATED HYPERPARAMETER REFERENCE",
            hp_ref_content,
            "",
            f"Generate script named: train_task{task_id}_round{self.round}.sh",
            f"Use round number: {self.round}",
            "Output ONLY the runnable bash script — no explanations, no markdown.",
        ])

        raw    = self._agent("step4_training_launcher", "training_launcher").run(context)
        script = AgentRunner.extract_bash_script(raw)

        script_name = f"train_task{task_id}_round{self.round}.sh"
        script_path = self.run_dir / script_name

        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o755)
        self.log.info("STEP 4 complete → %s", script_path)
        return script_path

    def step4b_generate_slurm(self, hp_ref_path: Path) -> Path:
        """
        Deterministically generate a SLURM array job script for the current round.
        Follows the same pattern as scripts/02_train_601.slurm in the repository.
        Saves to self.run_dir/train_taskXXX_roundN.slurm.
        Returns the path to the generated script.
        """
        self.log.info("── STEP 4b: Generate SLURM Script ──────────────────────")

        task_id  = self._task_id()
        round_n  = self.round
        job_name = f"nnunet_train_{task_id}_r{round_n}"
        log_stem = f"train_{task_id}_round{round_n}"

        slurm_content = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --output={log_stem}_fold%a_%j.out
#SBATCH --error={log_stem}_fold%a_%j.err

# Auto-generated SLURM training script — Round {round_n}
# Task:    {self.task}
# Network: 3d_fullres
# Trainer: nnUNetTrainerV2_configurable
# Plans:   nnUNetPlansv2.1
#
# Hyperparameter overrides are read automatically from:
#   {hp_ref_path}
#
# Submit all 5 folds:
#   sbatch --array=0-4 {log_stem}.slurm
#
# Resume interrupted folds:
#   sbatch --array=0-4 --export=ALL,CONTINUE=1 {log_stem}.slurm

set -euo pipefail

REPO_DIR="/scratch/work/tianmid1/nnUNet_cust"
DATA_BASE="/m/triton/scratch/elec/t41026-hintlab/tianmid1/data"

export nnUNet_raw_data_base="${{DATA_BASE}}"
export nnUNet_preprocessed="${{DATA_BASE}}/nnUNet_preprocessed"
export RESULTS_FOLDER="${{DATA_BASE}}/nnUNet_results"
export NNUNET_HP_REF="{hp_ref_path}"

mkdir -p "${{REPO_DIR}}/logs"

FOLD="${{SLURM_ARRAY_TASK_ID:-0}}"
CONTINUE="${{CONTINUE:-0}}"

NETWORK="3d_fullres"
TRAINER="nnUNetTrainerV2_configurable"
TASK="{task_id}"
PLANS="nnUNetPlansv2.1"

cd "${{REPO_DIR}}"

echo "=== nnUNet Training Round {round_n} ==="
echo "Network:     ${{NETWORK}}"
echo "Trainer:     ${{TRAINER}}"
echo "Task:        Task${{TASK}}_{self.task.split('_', 1)[-1] if '_' in self.task else self.task}"
echo "Round:       {round_n}"
echo "Fold:        ${{FOLD}}"
echo "Continue:    ${{CONTINUE}}"
echo "HP_REF:      ${{NNUNET_HP_REF}}"
echo "GPU:         ${{SLURM_STEP_GPUS:-${{CUDA_VISIBLE_DEVICES:-unset}}}}"
echo "Node:        ${{SLURM_NODELIST:-local}}"
echo "Started:     $(date)"
echo ""

CONTINUE_FLAG=""
if [ "${{CONTINUE}}" = "1" ]; then
    CONTINUE_FLAG="-c"
fi

uv run nnUNet_train \\
    "${{NETWORK}}" \\
    "${{TRAINER}}" \\
    "${{TASK}}" \\
    "${{FOLD}}" \\
    -p "${{PLANS}}" \\
    ${{CONTINUE_FLAG}}

echo ""
echo "Training fold ${{FOLD}} round {round_n} complete: $(date)"
"""

        slurm_name = f"train_task{task_id}_round{round_n}.slurm"
        slurm_path = self.run_dir / slurm_name
        slurm_path.write_text(slurm_content, encoding="utf-8")
        slurm_path.chmod(0o755)
        self.log.info("STEP 4b complete → %s", slurm_path)
        self.log.info(
            "  Submit with:  sbatch --array=0-4 %s", slurm_path
        )
        return slurm_path

    def step5_execute_training(self, script_path: Path, use_slurm: bool = False) -> None:
        """(Optional) Execute or submit the generated training script."""
        self.log.info("── STEP 5: Execute Training ─────────────────────────────")
        if use_slurm:
            slurm_script = script_path.with_suffix(".slurm")
            if not slurm_script.exists():
                raise FileNotFoundError(
                    f"step5: SLURM script not found: {slurm_script}. "
                    "Run with --slurm to generate it first."
                )
            cmd = ["sbatch", "--array=0-4", str(slurm_script)]
            self.log.info("Submitting: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.log.info(result.stdout.strip())
            if result.returncode != 0:
                self.log.error(result.stderr)
                raise RuntimeError(
                    f"step5: sbatch exited with code {result.returncode}"
                )
        else:
            self.log.info("Executing: bash %s", script_path)
            result = subprocess.run(["bash", str(script_path)], capture_output=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"step5: training script exited with code {result.returncode}"
                )
        self.log.info("STEP 5 complete")

    # ── full pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        test_summary_path: Path,
        val_summary_path: Path,
        hp_ref_path: Path,
        max_rounds: int = 1,
    ) -> None:
        """Execute the pipeline for *max_rounds* rounds."""
        start_round = self.round
        for i in range(max_rounds):
            self.log.info(
                "╔══ Round %d  (%d of %d) ══╗",
                self.round, i + 1, max_rounds,
            )
            try:
                self._run_once(test_summary_path, val_summary_path, hp_ref_path)
            except Exception as exc:
                self.log.error(
                    "Pipeline stopped at round %d: %s",
                    self.round, exc,
                    exc_info=True,
                )
                sys.exit(1)

            self.log.info("╚══ Round %d complete ══╝", self.round)
            self.round += 1

            # For multi-round runs: re-detect run_dir for next round
            if i + 1 < max_rounds:
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                self.run_dir = self.runs_dir / f"round_{self.round}_{ts}"
                self.run_dir.mkdir(parents=True, exist_ok=True)
                # Re-attach file handler for new run_dir
                self.log = _setup_logging(self.run_dir)

        completed = self.round - start_round
        self.log.info("=== Orchestrator finished | %d round(s) completed ===", completed)

    def _run_once(
        self,
        test_summary_path: Path,
        val_summary_path: Path,
        hp_ref_path: Path,
    ) -> None:
        # Carry checker result from a previous round into step 1
        previous_checker = self._find_previous_checker_result()
        if previous_checker:
            self.log.info("Using previous checker result: %s", previous_checker)

        # Step 1 — HP auto-tuning
        tuning_decision, tuning_path = self.step1_hp_autotuning(
            test_summary_path   = test_summary_path,
            val_summary_path    = val_summary_path,
            hp_ref_path         = hp_ref_path,
            checker_result_path = previous_checker,
        )

        # Step 2 — HP checker (result feeds into NEXT round's step 1)
        _checked, _checker_path = self.step2_hp_checker(hp_ref_path=hp_ref_path)

        # Step 3 — Apply tuning decisions
        updated_ref = self.step3_modify_hp(
            tuning_decision      = tuning_decision,
            tuning_decision_path = tuning_path,
            hp_ref_path          = hp_ref_path,
        )

        # Step 4 — Generate bash training script (saved to self.run_dir)
        script_path = self.step4_training_launcher(hp_ref_path=updated_ref)

        # Step 4b — Generate SLURM array script (optional)
        if self.generate_slurm:
            self.step4b_generate_slurm(hp_ref_path=updated_ref)

        self.log.info("Training script ready: %s", script_path)

        # Step 5 — (optional) execute / submit
        if self.auto_train:
            self.step5_execute_training(
                script_path, use_slurm=self.generate_slurm
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _default_test_summary(task: str, data_base: Path) -> Path:
    return (
        data_base / "nnUNet_results" / "nnUNet" / "3d_fullres" / task
        / "nnUNetTrainerV2_configurable__nnUNetPlansv2.1"
        / "summary.json"
    )


def _default_val_summary(task: str, data_base: Path) -> Path:
    return (
        data_base / "nnUNet_results" / "nnUNet" / "3d_fullres" / task
        / "nnUNetTrainerV2_configurable__nnUNetPlansv2.1"
        / "cv_niftis_postprocessed" / "summary.json"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="nnUNet hyperparameter auto-tuning multi-agent orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        default="Task601_TotalSegmentatorV1",
        help="nnUNet task name  (default: Task601_TotalSegmentatorV1)",
    )
    parser.add_argument(
        "--auto-train",
        action="store_true",
        help="Automatically execute / submit the generated training script (Step 5)",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help=(
            "Generate a SLURM array script (train_taskXXX_roundN.slurm) in addition "
            "to the bash script. If --auto-train is also set, submits via sbatch."
        ),
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1,
        metavar="N",
        help="Number of tuning rounds to run  (default: 1)",
    )
    parser.add_argument(
        "--test-summary",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to test results summary JSON  (auto-detected if omitted)",
    )
    parser.add_argument(
        "--val-summary",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to cross-validation summary JSON  (auto-detected if omitted)",
    )
    parser.add_argument(
        "--hp-ref",
        type=Path,
        default=DOC_DIR / "hyperparameter_reference.json",
        metavar="PATH",
        help="Path to hyperparameter_reference.json",
    )
    parser.add_argument(
        "--provider",
        default=PROVIDER_ANTHROPIC,
        choices=SUPPORTED_PROVIDERS,
        help=(
            f"LLM provider to use  (default: {PROVIDER_ANTHROPIC}). "
            f"'anthropic' requires ANTHROPIC_API_KEY; "
            f"'deepseek' requires DEEPSEEK_API_KEY and the openai package."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL_ID",
        help=(
            f"Model ID override. "
            f"Defaults: anthropic → {DEFAULT_MODEL}, "
            f"deepseek → {DEFAULT_DEEPSEEK_MODEL}"
        ),
    )
    parser.add_argument(
        "--doc-dir",
        type=Path,
        default=DOC_DIR,
        metavar="DIR",
        help=f"Documentation directory  (default: {DOC_DIR})",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR,
        metavar="DIR",
        help=f"Log directory  (default: {LOG_DIR})",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        metavar="DIR",
        help=f"Runs directory  (default: {RUNS_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── provider / API key validation ─────────────────────────────────────────
    if args.provider == PROVIDER_ANTHROPIC:
        if "ANTHROPIC_API_KEY" not in os.environ:
            print(
                "ERROR: ANTHROPIC_API_KEY is not set. "
                "Export it or use --provider deepseek.",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.provider == PROVIDER_DEEPSEEK:
        if not _OPENAI_AVAILABLE:
            print(
                "ERROR: 'openai' package is required for --provider deepseek. "
                "Install it with: pip install openai",
                file=sys.stderr,
            )
            sys.exit(1)
        if "DEEPSEEK_API_KEY" not in os.environ:
            print(
                "ERROR: DEEPSEEK_API_KEY is not set. "
                "Export it before running with --provider deepseek.",
                file=sys.stderr,
            )
            sys.exit(1)

    # ── resolve model default per provider ────────────────────────────────────
    if args.model is None:
        model = (
            DEFAULT_MODEL
            if args.provider == PROVIDER_ANTHROPIC
            else DEFAULT_DEEPSEEK_MODEL
        )
    else:
        model = args.model

    # Ensure directories exist
    for d in (args.doc_dir, args.log_dir, args.runs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Resolve data base path
    data_base = Path(
        os.environ.get(
            "nnUNet_raw_data_base",
            "/m/triton/scratch/elec/t41026-hintlab/tianmid1/data",
        )
    )

    # Resolve default summary paths when not supplied
    test_summary = args.test_summary or _default_test_summary(args.task, data_base)
    val_summary  = args.val_summary  or _default_val_summary(args.task, data_base)

    pipeline = PipelineManager(
        task           = args.task,
        doc_dir        = args.doc_dir,
        log_dir        = args.log_dir,
        runs_dir       = args.runs_dir,
        model          = model,
        provider       = args.provider,
        auto_train     = args.auto_train,
        generate_slurm = args.slurm,
    )

    pipeline.run(
        test_summary_path = test_summary,
        val_summary_path  = val_summary,
        hp_ref_path       = args.hp_ref,
        max_rounds        = args.max_rounds,
    )


if __name__ == "__main__":
    main()

"You are a senior nnUNet framework developer and medical imaging deep learning engineer,
specialized in code auditing and training system analysis.

Your task is to perform a COMPLETE hyperparameter audit of the nnUNet training framework.

Think step-by-step and build an internal parameter inventory before generating the final output.

---

Please systematically rescan and analyze ALL nnUNet training-related source code, including but not limited to:

- nnUNetTrainerV2_fast
- nnUNetTrainerV2
- nnUNetTrainer
- NetworkTrainer
- plans files
- data augmentation setup
- optimizer / loss / scheduler
- inference & validation configuration
- data loading pipeline
- architecture initialization
- deep supervision logic
- checkpointing system
- reproducibility and precision settings

Goal:
Produce a COMPLETE reference of ALL tunable hyperparameters involved in nnUNet training.

You MUST NOT rely on previous conversation summaries.
You MUST re-read the actual code.

---

## Scope Requirements

Include parameters from:

- custom trainer implementations
- inherited parent classes
- plans configuration
- hardcoded constants
- data augmentation
- inference configuration
- architecture definition
- loss functions
- optimization
- training loop control
- cross-validation
- GPU / precision / reproducibility settings

---

## Output TWO formats

---

### (A) Markdown Output (.md)

Generate a complete Markdown document:

# nnUNet Complete Hyperparameter Reference

## Category Name

| Parameter | Location | Default Value | Modifiable | Description |
|------------|------------|------------|------------|------------|

Requirements:

- grouped by category
- each parameter appears only once
- Location must include:
  - filename
  - class
  - function or line number (if identifiable)

---

### (B) JSON Output (.json)

Generate a JSON object with STRICT structure:

{
  ""parameter_name"": {
    ""location"": ""file → class → function or line"",
    ""default_value"": ""..."",
    ""value_range"": ""..."",
    ""category"": ""..."",
    ""description"": ""...""
  }
}

Rules:

- key = parameter name
- value MUST be a dictionary
- provide reasonable tunable ranges based on nnUNet best practices
- boolean parameters:
  ""value_range"": [true, false]
- if exact range unknown → provide engineering recommendation

---

## Critical Constraints

- MUST rescan code instead of summarizing
- MUST include inherited and hardcoded parameters
- NO partial lists
- OUTPUT MUST BE COMPLETE
- Markdown and JSON must be consistent
- JSON must be valid and parseable

---

Final output order:

1. Markdown document
2. JSON object"
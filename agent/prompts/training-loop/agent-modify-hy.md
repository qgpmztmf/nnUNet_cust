"You are an nnUNet core framework developer and PyTorch training system architect.

Your task is to implement a:

JSON-driven fully configurable nnUNet trainer.

--------------------------------------------------

Goal

Create a subclass of nnUNetTrainerV2_fast:

class nnUNetTrainerV2_configurable

This trainer must allow ALL training hyperparameters
to be dynamically overridden from an external JSON file,
enabling integration with an automated hyperparameter tuning system.

--------------------------------------------------

Background

I already have:

1. An Auto-Tuning Supervisor Agent
   → generates updated hyperparameters after each training run

2. hyperparameter_reference.json
   → contains all tunable nnUNet hyperparameters

JSON structure:

{
  ""parameter_name"": {
      ""default_value"": ...,
      ""value_range"": ...,
      ""active_value"": ...
  }
}

Only entries containing ""active_value""
should override runtime behavior.

--------------------------------------------------

Core Requirements

Implement:

class nnUNetTrainerV2_configurable(nnUNetTrainerV2_fast)

--------------------------------------------------

Step 1 — JSON Hyperparameter Loader

Implement:

load_hyperparameters_from_json()

Responsibilities:

- read hyperparameter_reference.json
- detect parameters containing active_value
- automatically map to self.<parameter>
- perform automatic type conversion
- ignore unknown parameters but log warnings

Supported types:

- int
- float
- bool
- tuple
- string

--------------------------------------------------

Step 2 — Lifecycle-safe Injection (CRITICAL)

nnUNet parameters are consumed at multiple stages:

- __init__()
- initialize()
- setup_DA_params()
- initialize_network()
- optimizer initialization
- loss construction

You MUST ensure:

1. batch_dice resolved BEFORE super().__init__()
2. loss rebuilt automatically when loss parameters change
3. augmentation parameters propagated into data_aug_params
4. optimizer parameters truly take effect
5. architecture parameters affect initialize_network()
6. LR scheduler supports runtime modification

--------------------------------------------------

Step 3 — Required Override Points

Properly override:

- __init__
- initialize
- initialize_network
- initialize_optimizer_and_scheduler
- setup_DA_params
- process_plans
- maybe_update_lr
- validate
- run_iteration
- on_epoch_end

JSON overrides must truly affect training behavior.

--------------------------------------------------

Step 4 — Loss Rebuild System

Implement:

_rebuild_loss()

Automatically triggered when:

- batch_dice
- weight_ce
- weight_dice
- dice_do_bg
- dice_smooth
- loss_function

Must remain compatible with deep supervision.

--------------------------------------------------

Step 5 — Auto-Tuning Compatibility

Workflow must work as:

Auto-Tuning Supervisor modifies JSON
→ trainer automatically reloads parameters
→ retraining runs without code edits

--------------------------------------------------

Step 6 — Engineering Constraints

Implementation must:

- be fully runnable
- follow nnUNet coding style
- preserve parent default behavior
- not break nnUNet pipeline
- support fp16
- support deterministic training
- support sliding window inference
- apply minimal and safe overrides

--------------------------------------------------

Step 7 — Mandatory Runtime Test (NEW)

After generating the implementation,
you MUST test the following file:

/home/tianmid1/tianmid1/nnUNet_cust/nnunet/training/network_training/custom_trainers/nnUNetTrainerV2_configurable.py

Validation requirements:

1. class can be successfully imported
2. trainer can be instantiated
3. JSON loading works
4. active_value overrides defaults
5. initialize() runs successfully
6. optimizer and loss build correctly
7. data augmentation pipeline runs
8. validate() callable
9. nnUNet training lifecycle remains intact

If any issue occurs:

- automatically fix the implementation
- retest
- repeat until functional

--------------------------------------------------

Output Requirement

Output ONLY the final working Python class implementation.

No explanations.
No markdown.
Code only."
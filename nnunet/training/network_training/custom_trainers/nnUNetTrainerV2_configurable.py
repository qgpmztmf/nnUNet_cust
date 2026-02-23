"""
nnUNetTrainerV2_configurable
============================
A fully JSON-configurable subclass of nnUNetTrainerV2_fast.

Every hyperparameter documented in
    documentation/hyperparameter_reference.json
is exposed as a self. attribute with a matching default.

To activate a parameter override, add an "active_value" key to the
corresponding entry in hyperparameter_reference.json:

    "oversample_foreground_percent": {
        ...existing fields...
        "active_value": 0.66
    }

The trainer calls load_hyperparameters_from_json() automatically at the
end of __init__.  You may also call it manually at any point BEFORE
initialize() is called (i.e., before training starts).

NOTE — moreDA hardcoded params
    gaussian_noise_p, gaussian_blur_sigma_range, gaussian_blur_p_per_sample,
    brightness_multiplicative_range, brightness_multiplicative_p,
    contrast_augmentation_p, simulate_lowres_zoom_range, and
    simulate_lowres_p_per_sample are stored as self. attributes for
    transparency but are hardcoded inside data_augmentation_moreDA.py.
    They have no effect unless you also edit that file.
    do_additive_brightness, additive_brightness_mu, additive_brightness_sigma
    ARE forwarded via data_aug_params and take effect normally.
"""

import json
import os
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, DC_and_topk_loss
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.network_training.custom_trainers.nnUNetTrainerV2_fast import nnUNetTrainerV2_fast
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


class nnUNetTrainerV2_configurable(nnUNetTrainerV2_fast):
    """
    JSON-driven configurable trainer.  See module docstring for usage.
    """

    # Default path — resolved relative to this file so it works from any cwd.
    HP_JSON_PATH: str = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..", "..", "documentation",
            "hyperparameter_reference.json",
        )
    )

    # ──────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
    ):
        # ── Step 1: read batch_dice from JSON BEFORE super().__init__() ──────
        # nnUNetTrainer.__init__() builds self.loss immediately using batch_dice,
        # so we must resolve it before the super() call.
        _pre = self._read_active_values(self.HP_JSON_PATH)
        batch_dice = bool(_pre.get("batch_dice", batch_dice))

        super().__init__(
            plans_file, fold, output_folder, dataset_directory,
            batch_dice, stage, unpack_data, deterministic, fp16,
        )

        # ── Step 2: declare ALL configurable attributes with defaults ─────────
        # Defaults match nnUNetTrainerV2_fast's actual runtime behaviour.

        # --- Training loop ---------------------------------------------------
        self.max_num_epochs            = 1000
        self.num_batches_per_epoch     = 32    # fast trainer default
        self.num_val_batches_per_epoch = 1     # fast trainer default
        self.also_val_in_tr_mode       = False
        self.patience                  = 50
        self.val_eval_criterion_alpha  = 0.9
        self.train_loss_MA_alpha       = 0.93
        self.train_loss_MA_eps         = 5e-4
        self.lr_threshold              = 1e-6

        # --- Optimizer -------------------------------------------------------
        self.initial_lr               = 1e-2
        self.weight_decay             = 3e-5
        self.sgd_momentum             = 0.99
        self.sgd_nesterov             = True
        self.poly_lr_exponent         = 0.9
        self.grad_clip_norm           = 12.0
        self.momentum_rescue_threshold = 0.95  # applied at epoch 100 if Dice==0

        # --- Loss function ---------------------------------------------------
        # batch_dice already set above (pre-super)
        self.dice_smooth              = 1e-5
        self.dice_do_bg               = False
        self.weight_dice              = 1.0
        self.weight_ce                = 1.0
        self.square_dice              = False
        self.log_dice                 = False
        self.topk_k                   = 10
        self.loss_function            = "DC_and_CE_loss"

        # --- Deep supervision ------------------------------------------------
        self.deep_supervision_enabled = True
        self.ds_lowest_levels_masked  = 1

        # --- Network architecture --------------------------------------------
        self.norm_op_type             = "InstanceNorm"  # "InstanceNorm" | "BatchNorm"
        self.norm_op_eps              = 1e-5
        self.norm_op_affine           = True
        self.dropout_p                = 0.0
        self.dropout_in_localization  = False
        self.nonlin_type              = "LeakyReLU"    # "LeakyReLU"|"ReLU"|"ELU"|"GELU"
        self.nonlin_negative_slope    = 1e-2
        self.convolutional_pooling    = True
        self.convolutional_upsampling = True
        self.seg_output_use_bias      = False
        self.he_init_neg_slope        = 1e-2

        # --- Data augmentation -----------------------------------------------
        # Spatial
        self.aug_do_elastic              = False
        self.aug_elastic_deform_alpha    = (0.0, 900.0)
        self.aug_elastic_deform_sigma    = (9.0, 13.0)
        self.aug_p_eldef                 = 0.2
        self.aug_do_scaling              = True
        self.aug_scale_range             = (0.7, 1.4)
        self.aug_p_scale                 = 0.2
        self.aug_independent_scale_axes  = False
        self.aug_do_rotation             = True
        self.aug_rotation_x              = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.aug_rotation_y              = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.aug_rotation_z              = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.aug_rotation_p_per_axis     = 1.0
        self.aug_p_rot                   = 0.2
        # Intensity (via params dict → data_augmentation_moreDA.py)
        self.aug_do_gamma                = True
        self.aug_gamma_range             = (0.7, 1.5)
        self.aug_gamma_retain_stats      = True
        self.aug_p_gamma                 = 0.3
        self.aug_gamma_inverted_p        = 0.1   # hardcoded in moreDA (see module note)
        self.aug_do_additive_brightness  = False  # forwarded via params dict
        self.aug_additive_brightness_mu  = 0.0
        self.aug_additive_brightness_sigma = 0.1
        # Mirror / misc
        self.aug_do_mirror               = False  # fast trainer default
        self.aug_mirror_axes             = (0, 1, 2)
        self.aug_dummy_2D                = False
        self.aug_border_mode_data        = "constant"
        self.aug_num_cached_per_thread   = 2
        # moreDA hardcoded (stored for transparency; require moreDA.py edits to activate)
        self.aug_gaussian_noise_p        = 0.1
        self.aug_gaussian_blur_sigma     = (0.5, 1.0)
        self.aug_gaussian_blur_p         = 0.2
        self.aug_brightness_range        = (0.75, 1.25)
        self.aug_brightness_p            = 0.15
        self.aug_contrast_p              = 0.15
        self.aug_lowres_zoom_range       = (0.5, 1.0)
        self.aug_lowres_p                = 0.25

        # --- Data loading ----------------------------------------------------
        self.oversample_foreground_percent = 0.33
        self.pin_memory                    = True

        # --- Checkpointing ---------------------------------------------------
        self.save_every                    = 50
        self.save_latest_only              = True
        self.save_intermediate_checkpoints = True
        self.save_best_checkpoint          = True
        self.save_final_checkpoint         = True

        # --- LR scheduler / inference ----------------------------------------
        self.lr_scheduler_eps              = 1e-3
        self.lr_scheduler_patience         = 30
        self.inference_pad_border_mode     = "constant"
        self.inference_pad_kwargs          = {"constant_values": 0}

        # --- Fast-trainer batch size override --------------------------------
        self.batch_size_forced_fast        = 16

        # ── Step 3: apply all active_value overrides from JSON ───────────────
        self.load_hyperparameters_from_json()

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════

    def load_hyperparameters_from_json(self, path: str = None) -> None:
        """
        Read every entry that has an "active_value" key from the reference
        JSON and apply it to this trainer instance.

        Parameters
        ----------
        path : str, optional
            Path to a hyperparameter reference JSON.
            Defaults to self.HP_JSON_PATH.

        Notes
        -----
        * Call this before initialize() / training start.
        * batch_dice changes trigger automatic loss re-initialisation.
        * moreDA hardcoded params (gaussian_noise_p, etc.) are stored on self
          but have no training effect without editing data_augmentation_moreDA.py.
        """
        path = path or self.HP_JSON_PATH
        active = self._read_active_values(path)
        if not active:
            return

        self.print_to_log_file(
            f"[configurable] Loading {len(active)} active override(s) from:\n  {path}"
        )

        _loss_dirty = False  # track whether the loss needs rebuilding

        # ── Training loop ────────────────────────────────────────────────────
        if "max_num_epochs" in active:
            self.max_num_epochs = int(active["max_num_epochs"])
        if "num_batches_per_epoch" in active:
            self.num_batches_per_epoch = int(active["num_batches_per_epoch"])
        if "num_val_batches_per_epoch" in active:
            self.num_val_batches_per_epoch = int(active["num_val_batches_per_epoch"])
        if "also_val_in_tr_mode" in active:
            self.also_val_in_tr_mode = bool(active["also_val_in_tr_mode"])
        if "patience" in active:
            self.patience = active["patience"]          # None disables early stop
        if "val_eval_criterion_alpha" in active:
            self.val_eval_criterion_alpha = float(active["val_eval_criterion_alpha"])
        if "train_loss_MA_alpha" in active:
            self.train_loss_MA_alpha = float(active["train_loss_MA_alpha"])
        if "train_loss_MA_eps" in active:
            self.train_loss_MA_eps = float(active["train_loss_MA_eps"])
        if "lr_threshold" in active:
            self.lr_threshold = float(active["lr_threshold"])

        # ── Optimizer ────────────────────────────────────────────────────────
        if "initial_lr_nnUNetTrainerV2" in active:
            self.initial_lr = float(active["initial_lr_nnUNetTrainerV2"])
        if "weight_decay" in active:
            self.weight_decay = float(active["weight_decay"])
        if "sgd_momentum" in active:
            self.sgd_momentum = float(active["sgd_momentum"])
        if "sgd_nesterov" in active:
            self.sgd_nesterov = bool(active["sgd_nesterov"])
        if "poly_lr_exponent" in active:
            self.poly_lr_exponent = float(active["poly_lr_exponent"])
        if "grad_clip_norm" in active:
            self.grad_clip_norm = float(active["grad_clip_norm"])
        if "momentum_rescue_threshold" in active:
            self.momentum_rescue_threshold = float(active["momentum_rescue_threshold"])

        # ── Loss function ────────────────────────────────────────────────────
        if "batch_dice" in active:
            new_bd = bool(active["batch_dice"])
            if new_bd != self.batch_dice:
                self.batch_dice = new_bd
                _loss_dirty = True
        if "dice_smooth" in active:
            self.dice_smooth = float(active["dice_smooth"]); _loss_dirty = True
        if "dice_do_bg" in active:
            self.dice_do_bg = bool(active["dice_do_bg"]); _loss_dirty = True
        if "weight_dice" in active:
            self.weight_dice = float(active["weight_dice"]); _loss_dirty = True
        if "weight_ce" in active:
            self.weight_ce = float(active["weight_ce"]); _loss_dirty = True
        if "square_dice" in active:
            self.square_dice = bool(active["square_dice"]); _loss_dirty = True
        if "log_dice" in active:
            self.log_dice = bool(active["log_dice"]); _loss_dirty = True
        if "loss_function" in active:
            self.loss_function = str(active["loss_function"]); _loss_dirty = True
        if "topk_k" in active:
            self.topk_k = int(active["topk_k"]); _loss_dirty = True
        if _loss_dirty:
            self._rebuild_loss()

        # ── Deep supervision ─────────────────────────────────────────────────
        if "deep_supervision_enabled" in active:
            self.deep_supervision_enabled = bool(active["deep_supervision_enabled"])
        if "ds_lowest_levels_masked" in active:
            self.ds_lowest_levels_masked = int(active["ds_lowest_levels_masked"])

        # ── Network architecture ─────────────────────────────────────────────
        if "norm_op" in active:
            self.norm_op_type = str(active["norm_op"])
        if "norm_op_eps" in active:
            self.norm_op_eps = float(active["norm_op_eps"])
        if "norm_op_affine" in active:
            self.norm_op_affine = bool(active["norm_op_affine"])
        if "dropout_p" in active:
            self.dropout_p = float(active["dropout_p"])
        if "dropout_in_localization" in active:
            self.dropout_in_localization = bool(active["dropout_in_localization"])
        if "nonlin" in active:
            self.nonlin_type = str(active["nonlin"])
        if "nonlin_negative_slope" in active:
            self.nonlin_negative_slope = float(active["nonlin_negative_slope"])
        if "convolutional_pooling" in active:
            self.convolutional_pooling = bool(active["convolutional_pooling"])
        if "convolutional_upsampling" in active:
            self.convolutional_upsampling = bool(active["convolutional_upsampling"])
        if "seg_output_use_bias" in active:
            self.seg_output_use_bias = bool(active["seg_output_use_bias"])
        if "he_init_neg_slope" in active:
            self.he_init_neg_slope = float(active["he_init_neg_slope"])

        # ── Data augmentation — spatial ──────────────────────────────────────
        if "do_elastic" in active:
            self.aug_do_elastic = bool(active["do_elastic"])
        if "elastic_deform_alpha" in active:
            self.aug_elastic_deform_alpha = tuple(active["elastic_deform_alpha"])
        if "elastic_deform_sigma" in active:
            self.aug_elastic_deform_sigma = tuple(active["elastic_deform_sigma"])
        if "p_eldef" in active:
            self.aug_p_eldef = float(active["p_eldef"])
        if "do_scaling" in active:
            self.aug_do_scaling = bool(active["do_scaling"])
        if "scale_range" in active:
            self.aug_scale_range = tuple(active["scale_range"])
        if "p_scale" in active:
            self.aug_p_scale = float(active["p_scale"])
        if "independent_scale_factor_for_each_axis" in active:
            self.aug_independent_scale_axes = bool(active["independent_scale_factor_for_each_axis"])
        if "do_rotation" in active:
            self.aug_do_rotation = bool(active["do_rotation"])
        if "rotation_x_V2" in active:
            self.aug_rotation_x = tuple(active["rotation_x_V2"])
        if "rotation_y_default" in active:
            self.aug_rotation_y = tuple(active["rotation_y_default"])
        if "rotation_z_default" in active:
            self.aug_rotation_z = tuple(active["rotation_z_default"])
        if "rotation_p_per_axis" in active:
            self.aug_rotation_p_per_axis = float(active["rotation_p_per_axis"])
        if "p_rot" in active:
            self.aug_p_rot = float(active["p_rot"])

        # ── Data augmentation — intensity ────────────────────────────────────
        if "do_gamma" in active:
            self.aug_do_gamma = bool(active["do_gamma"])
        if "gamma_range" in active:
            self.aug_gamma_range = tuple(active["gamma_range"])
        if "gamma_retain_stats" in active:
            self.aug_gamma_retain_stats = bool(active["gamma_retain_stats"])
        if "p_gamma" in active:
            self.aug_p_gamma = float(active["p_gamma"])
        if "gamma_inverted_p" in active:
            self.aug_gamma_inverted_p = float(active["gamma_inverted_p"])
        if "do_additive_brightness" in active:
            self.aug_do_additive_brightness = bool(active["do_additive_brightness"])
        if "additive_brightness_mu" in active:
            self.aug_additive_brightness_mu = float(active["additive_brightness_mu"])
        if "additive_brightness_sigma" in active:
            self.aug_additive_brightness_sigma = float(active["additive_brightness_sigma"])
        # moreDA hardcoded — stored for transparency only
        if "gaussian_noise_p" in active:
            self.aug_gaussian_noise_p = float(active["gaussian_noise_p"])
        if "gaussian_blur_sigma_range" in active:
            self.aug_gaussian_blur_sigma = tuple(active["gaussian_blur_sigma_range"])
        if "gaussian_blur_p_per_sample" in active:
            self.aug_gaussian_blur_p = float(active["gaussian_blur_p_per_sample"])
        if "brightness_multiplicative_range" in active:
            self.aug_brightness_range = tuple(active["brightness_multiplicative_range"])
        if "brightness_multiplicative_p" in active:
            self.aug_brightness_p = float(active["brightness_multiplicative_p"])
        if "contrast_augmentation_p" in active:
            self.aug_contrast_p = float(active["contrast_augmentation_p"])
        if "simulate_lowres_zoom_range" in active:
            self.aug_lowres_zoom_range = tuple(active["simulate_lowres_zoom_range"])
        if "simulate_lowres_p_per_sample" in active:
            self.aug_lowres_p = float(active["simulate_lowres_p_per_sample"])

        # ── Data augmentation — misc ─────────────────────────────────────────
        if "do_mirror" in active:
            self.aug_do_mirror = bool(active["do_mirror"])
        if "mirror_axes" in active:
            self.aug_mirror_axes = tuple(active["mirror_axes"])
        if "dummy_2D" in active:
            self.aug_dummy_2D = bool(active["dummy_2D"])
        if "border_mode_data" in active:
            self.aug_border_mode_data = str(active["border_mode_data"])
        if "num_cached_per_thread" in active:
            self.aug_num_cached_per_thread = int(active["num_cached_per_thread"])

        # ── Data loading ─────────────────────────────────────────────────────
        if "oversample_foreground_percent" in active:
            self.oversample_foreground_percent = float(active["oversample_foreground_percent"])
        if "pin_memory" in active:
            self.pin_memory = bool(active["pin_memory"])

        # ── Checkpointing ────────────────────────────────────────────────────
        if "save_every" in active:
            self.save_every = int(active["save_every"])
        if "save_latest_only" in active:
            self.save_latest_only = bool(active["save_latest_only"])
        if "save_intermediate_checkpoints" in active:
            self.save_intermediate_checkpoints = bool(active["save_intermediate_checkpoints"])
        if "save_best_checkpoint" in active:
            self.save_best_checkpoint = bool(active["save_best_checkpoint"])
        if "save_final_checkpoint" in active:
            self.save_final_checkpoint = bool(active["save_final_checkpoint"])

        # ── Inference & validation ───────────────────────────────────────────
        if "lr_scheduler_eps" in active:
            self.lr_scheduler_eps = float(active["lr_scheduler_eps"])
        if "lr_scheduler_patience" in active:
            self.lr_scheduler_patience = int(active["lr_scheduler_patience"])
        if "inference_pad_border_mode" in active:
            self.inference_pad_border_mode = str(active["inference_pad_border_mode"])
        if "inference_pad_constant_value" in active:
            self.inference_pad_kwargs = {"constant_values": float(active["inference_pad_constant_value"])}

        # ── Fast-trainer batch size ──────────────────────────────────────────
        if "batch_size_forced_fast" in active:
            self.batch_size_forced_fast = int(active["batch_size_forced_fast"])

        self.print_to_log_file("[configurable] Hyperparameter overrides applied.")

    # ══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _read_active_values(path: str) -> dict:
        """
        Parse the reference JSON and return {param_name: active_value} for
        every entry that contains an "active_value" key.
        Returns an empty dict if the file does not exist.
        """
        if not os.path.isfile(path):
            return {}
        with open(path, "r") as fh:
            raw = json.load(fh)
        return {k: v["active_value"] for k, v in raw.items() if "active_value" in v}

    def _rebuild_loss(self) -> None:
        """
        Re-instantiate self.loss from the current loss-related attributes.

        If initialize() has already been called (self.was_initialized=True),
        the new base loss is re-wrapped with MultipleOutputLoss2 so that deep
        supervision continues to work.
        """
        soft_dice_kwargs = {
            "batch_dice": self.batch_dice,
            "smooth":     self.dice_smooth,
            "do_bg":      self.dice_do_bg,
        }

        if self.loss_function == "DC_and_topk_loss":
            base_loss = DC_and_topk_loss(
                soft_dice_kwargs,
                {"k": self.topk_k},
                square_dice=self.square_dice,
            )
        else:  # default: DC_and_CE_loss
            base_loss = DC_and_CE_loss(
                soft_dice_kwargs,
                {},
                weight_ce=self.weight_ce,
                weight_dice=self.weight_dice,
                square_dice=self.square_dice,
                log_dice=self.log_dice,
            )

        if getattr(self, "was_initialized", False) and self.ds_loss_weights is not None:
            # Re-wrap for deep supervision
            self.loss = MultipleOutputLoss2(base_loss, self.ds_loss_weights)
        else:
            self.loss = base_loss

    # ══════════════════════════════════════════════════════════════════════════
    # Overridden lifecycle methods
    # ══════════════════════════════════════════════════════════════════════════

    def initialize(self, training: bool = True, force_load_plans: bool = False) -> None:
        """
        Mirrors nnUNetTrainerV2.initialize() but uses:
          - self.ds_lowest_levels_masked  (configurable, default 1)
          - self.deep_supervision_enabled (configurable, default True)
        """
        if self.was_initialized:
            self.print_to_log_file("self.was_initialized is True, skipping initialize()")
            return

        maybe_mkdir_p(self.output_folder)

        if force_load_plans or self.plans is None:
            self.load_plans_file()

        self.process_plans(self.plans)
        self.setup_DA_params()

        # ── Deep supervision weights ─────────────────────────────────────────
        net_numpool = len(self.net_num_pool_op_kernel_sizes)
        weights = np.array([1 / (2 ** i) for i in range(net_numpool)])
        mask = np.array(
            [True]
            + [
                True if i < net_numpool - self.ds_lowest_levels_masked else False
                for i in range(1, net_numpool)
            ]
        )
        weights[~mask] = 0.0
        weights = weights / weights.sum()
        self.ds_loss_weights = weights

        # Wrap the base loss (already built/rebuilt in __init__) for DS
        self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

        # ── Data generators ──────────────────────────────────────────────────
        self.folder_with_preprocessed_data = join(
            self.dataset_directory,
            self.plans["data_identifier"] + "_stage%d" % self.stage,
        )

        if training:
            self.dl_tr, self.dl_val = self.get_basic_generators()
            if self.unpack_data:
                print("unpacking dataset")
                unpack_dataset(self.folder_with_preprocessed_data)
                print("done")

            self.tr_gen, self.val_gen = get_moreDA_augmentation(
                self.dl_tr,
                self.dl_val,
                self.data_aug_params["patch_size_for_spatialtransform"],
                self.data_aug_params,
                deep_supervision_scales=self.deep_supervision_scales,
                pin_memory=self.pin_memory,
                use_nondetMultiThreadedAugmenter=False,
            )
            self.print_to_log_file(
                "TRAINING KEYS:\n %s" % str(self.dataset_tr.keys()),
                also_print_to_console=False,
            )
            self.print_to_log_file(
                "VALIDATION KEYS:\n %s" % str(self.dataset_val.keys()),
                also_print_to_console=False,
            )

        self.initialize_network()
        self.initialize_optimizer_and_scheduler()

        assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        self.was_initialized = True

    # ──────────────────────────────────────────────────────────────────────────

    def initialize_network(self) -> None:
        """Build Generic_UNet from configurable architecture attributes."""
        if self.threeD:
            conv_op    = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op    = nn.BatchNorm3d if self.norm_op_type == "BatchNorm" else nn.InstanceNorm3d
        else:
            conv_op    = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op    = nn.BatchNorm2d if self.norm_op_type == "BatchNorm" else nn.InstanceNorm2d

        norm_op_kwargs    = {"eps": self.norm_op_eps, "affine": self.norm_op_affine}
        dropout_op_kwargs = {"p": self.dropout_p, "inplace": True}

        _nonlin_map = {
            "LeakyReLU": nn.LeakyReLU,
            "ReLU":      nn.ReLU,
            "ELU":       nn.ELU,
            "GELU":      nn.GELU,
        }
        net_nonlin = _nonlin_map.get(self.nonlin_type, nn.LeakyReLU)

        # Build nonlin kwargs — only LeakyReLU uses negative_slope
        if self.nonlin_type == "LeakyReLU":
            net_nonlin_kwargs = {"negative_slope": self.nonlin_negative_slope, "inplace": True}
        elif self.nonlin_type in ("ReLU", "ELU"):
            net_nonlin_kwargs = {"inplace": True}
        else:  # GELU etc. — no inplace
            net_nonlin_kwargs = {}

        self.network = Generic_UNet(
            self.num_input_channels,
            self.base_num_features,
            self.num_classes,
            len(self.net_num_pool_op_kernel_sizes),  # num_pool
            self.conv_per_stage,
            2,                                        # feat_map_mul_on_downscale
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            net_nonlin,
            net_nonlin_kwargs,
            self.deep_supervision_enabled,            # do_ds
            self.dropout_in_localization,
            lambda x: x,                              # final_nonlin (softmax at inference)
            InitWeights_He(self.he_init_neg_slope),
            self.net_num_pool_op_kernel_sizes,
            self.net_conv_kernel_sizes,
            False,                                    # upscale_logits
            self.convolutional_pooling,
            self.convolutional_upsampling,
        )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    # ──────────────────────────────────────────────────────────────────────────

    def initialize_optimizer_and_scheduler(self) -> None:
        """Use configurable SGD momentum / nesterov."""
        assert self.network is not None, "Call initialize_network() first"
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=self.sgd_momentum,
            nesterov=self.sgd_nesterov,
        )
        self.lr_scheduler = None

    # ──────────────────────────────────────────────────────────────────────────

    def setup_DA_params(self) -> None:
        """
        Call the parent chain (sets base defaults and disables mirror for fast
        trainer), then override with all configurable aug attributes.
        """
        super().setup_DA_params()  # → nnUNetTrainerV2_fast → nnUNetTrainerV2

        # Spatial transforms (consumed by get_moreDA_augmentation via params dict)
        self.data_aug_params["do_elastic"]                             = self.aug_do_elastic
        self.data_aug_params["elastic_deform_alpha"]                   = self.aug_elastic_deform_alpha
        self.data_aug_params["elastic_deform_sigma"]                   = self.aug_elastic_deform_sigma
        self.data_aug_params["p_eldef"]                                = self.aug_p_eldef
        self.data_aug_params["do_scaling"]                             = self.aug_do_scaling
        self.data_aug_params["scale_range"]                            = self.aug_scale_range
        self.data_aug_params["p_scale"]                                = self.aug_p_scale
        self.data_aug_params["independent_scale_factor_for_each_axis"] = self.aug_independent_scale_axes
        self.data_aug_params["do_rotation"]                            = self.aug_do_rotation
        self.data_aug_params["rotation_x"]                             = self.aug_rotation_x
        self.data_aug_params["rotation_y"]                             = self.aug_rotation_y
        self.data_aug_params["rotation_z"]                             = self.aug_rotation_z
        self.data_aug_params["rotation_p_per_axis"]                    = self.aug_rotation_p_per_axis
        self.data_aug_params["p_rot"]                                  = self.aug_p_rot
        self.data_aug_params["border_mode_data"]                       = self.aug_border_mode_data
        self.data_aug_params["dummy_2D"]                               = self.aug_dummy_2D

        # Intensity transforms
        self.data_aug_params["do_gamma"]                               = self.aug_do_gamma
        self.data_aug_params["gamma_range"]                            = self.aug_gamma_range
        self.data_aug_params["gamma_retain_stats"]                     = self.aug_gamma_retain_stats
        self.data_aug_params["p_gamma"]                                = self.aug_p_gamma
        # Additive brightness forwarded to moreDA via params
        self.data_aug_params["do_additive_brightness"]                 = self.aug_do_additive_brightness
        self.data_aug_params["additive_brightness_mu"]                 = self.aug_additive_brightness_mu
        self.data_aug_params["additive_brightness_sigma"]              = self.aug_additive_brightness_sigma
        self.data_aug_params["additive_brightness_p_per_sample"]       = 0.15  # moreDA default
        self.data_aug_params["additive_brightness_p_per_channel"]      = 0.5

        # Mirror
        self.data_aug_params["do_mirror"]                              = self.aug_do_mirror
        self.data_aug_params["mirror_axes"]                            = self.aug_mirror_axes

        # Misc
        self.data_aug_params["num_cached_per_thread"]                  = self.aug_num_cached_per_thread

    # ──────────────────────────────────────────────────────────────────────────

    def process_plans(self, plans) -> None:
        """Override batch size via self.batch_size_forced_fast."""
        super().process_plans(plans)
        self.batch_size = self.batch_size_forced_fast

    # ──────────────────────────────────────────────────────────────────────────

    def maybe_update_lr(self, epoch=None) -> None:
        """Use configurable poly_lr_exponent."""
        ep = (self.epoch + 1) if epoch is None else epoch
        self.optimizer.param_groups[0]["lr"] = poly_lr(
            ep, self.max_num_epochs, self.initial_lr, self.poly_lr_exponent
        )
        self.print_to_log_file(
            "lr:", np.round(self.optimizer.param_groups[0]["lr"], decimals=6)
        )

    # ──────────────────────────────────────────────────────────────────────────

    def on_epoch_end(self):
        """Use configurable momentum_rescue_threshold at epoch 100."""
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        if self.epoch == 100 and self.all_val_eval_metrics[-1] == 0:
            self.optimizer.param_groups[0]["momentum"] = self.momentum_rescue_threshold
            self.network.apply(InitWeights_He(self.he_init_neg_slope))
            self.print_to_log_file(
                f"At epoch 100, foreground Dice is 0. Reducing momentum to "
                f"{self.momentum_rescue_threshold} and reinitialising weights."
            )
        return continue_training

    # ──────────────────────────────────────────────────────────────────────────

    def validate(
        self,
        do_mirroring: bool = True,
        use_sliding_window: bool = True,
        step_size: float = 0.5,
        save_softmax: bool = True,
        use_gaussian: bool = True,
        overwrite: bool = True,
        validation_folder_name: str = "validation_raw",
        debug: bool = False,
        all_in_gpu: bool = False,
        segmentation_export_kwargs: dict = None,
        run_postprocessing_on_folds: bool = True,
    ):
        """
        Use self.aug_do_mirror to decide TTA mirroring (overrides the
        hard-coded False in nnUNetTrainerV2_fast).
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        # nnUNetTrainer.validate (grandparent) is the one that uses do_mirroring
        from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
        ret = nnUNetTrainer.validate(
            self,
            do_mirroring=self.aug_do_mirror,  # respect configurable setting
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            save_softmax=save_softmax,
            use_gaussian=use_gaussian,
            overwrite=overwrite,
            validation_folder_name=validation_folder_name,
            debug=debug,
            all_in_gpu=all_in_gpu,
            segmentation_export_kwargs=segmentation_export_kwargs,
            run_postprocessing_on_folds=run_postprocessing_on_folds,
        )
        self.network.do_ds = ds
        return ret

    # ──────────────────────────────────────────────────────────────────────────

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """Use configurable grad_clip_norm."""
        data_dict = next(data_generator)
        data      = maybe_to_torch(data_dict["data"])
        target    = maybe_to_torch(data_dict["target"])

        if torch.cuda.is_available():
            data   = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)
            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)
            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        return l.detach().cpu().numpy()

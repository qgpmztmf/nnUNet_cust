"""
Tests for nnUNetTrainerV2_configurable
======================================
Run from the nnUNet_cust project root:

    uv run python -m pytest tests/test_nnUNetTrainerV2_configurable.py -v

Design notes
------------
* print_to_log_file is silenced inside _make_trainer() so no log files are
  ever created on disk.
* Each test class creates its own tempdir in setUp/setUpClass and removes it
  in tearDown/tearDownClass.
* Network, optimizer and data-generator interactions are mocked only where the
  method under test strictly requires them.
"""

import json
import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

_SILENCE = "nnunet.training.network_training.network_trainer.NetworkTrainer.print_to_log_file"


def _make_trainer(tmpdir: str, extra_json: dict = None, batch_dice: bool = True):
    """
    Instantiate nnUNetTrainerV2_configurable with minimal stub arguments.

    Parameters
    ----------
    tmpdir     : working directory (used as output_folder)
    extra_json : {param_name: active_value} entries to inject into a copy of
                 hyperparameter_reference.json before instantiation
    batch_dice : constructor-level default (may be overridden by extra_json)
    """
    from nnunet.training.network_training.custom_trainers.nnUNetTrainerV2_configurable import (
        nnUNetTrainerV2_configurable as Cls,
    )

    plans_stub = os.path.join(tmpdir, "plans_stub.pkl")
    open(plans_stub, "wb").close()

    ctx_json = None  # holds the patched HP_JSON_PATH when extra_json is given

    if extra_json:
        with open(Cls.HP_JSON_PATH) as fh:
            ref = json.load(fh)
        for key, val in extra_json.items():
            if key in ref:
                ref[key]["active_value"] = val
            else:
                ref[key] = {"active_value": val}
        tmp_json = os.path.join(tmpdir, "hp_patched.json")
        with open(tmp_json, "w") as fh:
            json.dump(ref, fh)
        ctx_json = patch.object(Cls, "HP_JSON_PATH", tmp_json)

    with patch(_SILENCE, return_value=None):
        if ctx_json:
            with ctx_json:
                t = Cls(
                    plans_file=plans_stub,
                    fold=0,
                    output_folder=tmpdir,
                    dataset_directory=None,
                    batch_dice=batch_dice,
                )
        else:
            t = Cls(
                plans_file=plans_stub,
                fold=0,
                output_folder=tmpdir,
                dataset_directory=None,
                batch_dice=batch_dice,
            )

    return t


# ══════════════════════════════════════════════════════════════════════════════
# 1. Static helper: _read_active_values
# ══════════════════════════════════════════════════════════════════════════════
class TestReadActiveValues(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, data: dict) -> str:
        path = os.path.join(self.tmpdir, "hp.json")
        with open(path, "w") as fh:
            json.dump(data, fh)
        return path

    def _fn(self):
        from nnunet.training.network_training.custom_trainers.nnUNetTrainerV2_configurable import (
            nnUNetTrainerV2_configurable,
        )
        return nnUNetTrainerV2_configurable._read_active_values

    def test_missing_file_returns_empty(self):
        self.assertEqual(self._fn()("/nonexistent/path.json"), {})

    def test_no_active_values_returns_empty(self):
        path = self._write({"p": {"default_value": "1"}, "q": {"default_value": "2"}})
        self.assertEqual(self._fn()(path), {})

    def test_extracts_only_active_values(self):
        path = self._write({
            "param_a": {"default_value": "1", "active_value": 42},
            "param_b": {"default_value": "2"},
            "param_c": {"default_value": "3", "active_value": False},
        })
        self.assertEqual(self._fn()(path), {"param_a": 42, "param_c": False})

    def test_all_value_types(self):
        path = self._write({
            "i": {"active_value": 100},
            "f": {"active_value": 0.007},
            "b": {"active_value": False},
            "l": {"active_value": [0.7, 1.4]},
            "s": {"active_value": "BatchNorm"},
        })
        r = self._fn()(path)
        self.assertEqual(r["i"], 100)
        self.assertAlmostEqual(r["f"], 0.007)
        self.assertFalse(r["b"])
        self.assertEqual(r["l"], [0.7, 1.4])
        self.assertEqual(r["s"], "BatchNorm")

    def test_live_reference_json_ships_without_active_values(self):
        """The committed JSON must not contain any active_value keys."""
        from nnunet.training.network_training.custom_trainers.nnUNetTrainerV2_configurable import (
            nnUNetTrainerV2_configurable as Cls,
        )
        result = Cls._read_active_values(Cls.HP_JSON_PATH)
        self.assertEqual(result, {},
                         "hyperparameter_reference.json must ship with no active_value entries")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Default attribute values after plain instantiation
# ══════════════════════════════════════════════════════════════════════════════
class TestDefaultAttributes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.t = _make_trainer(cls.tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    # Training loop
    def test_max_num_epochs(self):             self.assertEqual(self.t.max_num_epochs, 1000)
    def test_num_batches_per_epoch(self):      self.assertEqual(self.t.num_batches_per_epoch, 32)
    def test_num_val_batches_per_epoch(self):  self.assertEqual(self.t.num_val_batches_per_epoch, 1)
    def test_also_val_in_tr_mode(self):        self.assertFalse(self.t.also_val_in_tr_mode)
    def test_patience(self):                   self.assertEqual(self.t.patience, 50)
    def test_val_eval_criterion_alpha(self):   self.assertAlmostEqual(self.t.val_eval_criterion_alpha, 0.9)
    def test_train_loss_MA_alpha(self):        self.assertAlmostEqual(self.t.train_loss_MA_alpha, 0.93)
    def test_train_loss_MA_eps(self):          self.assertAlmostEqual(self.t.train_loss_MA_eps, 5e-4)
    def test_lr_threshold(self):               self.assertAlmostEqual(self.t.lr_threshold, 1e-6)

    # Optimizer
    def test_initial_lr(self):                 self.assertAlmostEqual(self.t.initial_lr, 1e-2)
    def test_weight_decay(self):               self.assertAlmostEqual(self.t.weight_decay, 3e-5)
    def test_sgd_momentum(self):               self.assertAlmostEqual(self.t.sgd_momentum, 0.99)
    def test_sgd_nesterov(self):               self.assertTrue(self.t.sgd_nesterov)
    def test_poly_lr_exponent(self):           self.assertAlmostEqual(self.t.poly_lr_exponent, 0.9)
    def test_grad_clip_norm(self):             self.assertAlmostEqual(self.t.grad_clip_norm, 12.0)
    def test_momentum_rescue_threshold(self):  self.assertAlmostEqual(self.t.momentum_rescue_threshold, 0.95)

    # Loss
    def test_batch_dice_default(self):         self.assertTrue(self.t.batch_dice)
    def test_dice_smooth(self):                self.assertAlmostEqual(self.t.dice_smooth, 1e-5)
    def test_dice_do_bg(self):                 self.assertFalse(self.t.dice_do_bg)
    def test_weight_dice(self):                self.assertAlmostEqual(self.t.weight_dice, 1.0)
    def test_weight_ce(self):                  self.assertAlmostEqual(self.t.weight_ce, 1.0)
    def test_square_dice(self):                self.assertFalse(self.t.square_dice)
    def test_log_dice(self):                   self.assertFalse(self.t.log_dice)
    def test_topk_k(self):                     self.assertEqual(self.t.topk_k, 10)
    def test_loss_function(self):              self.assertEqual(self.t.loss_function, "DC_and_CE_loss")

    # Deep supervision
    def test_deep_supervision_enabled(self):   self.assertTrue(self.t.deep_supervision_enabled)
    def test_ds_lowest_levels_masked(self):    self.assertEqual(self.t.ds_lowest_levels_masked, 1)

    # Network architecture
    def test_norm_op_type(self):               self.assertEqual(self.t.norm_op_type, "InstanceNorm")
    def test_norm_op_eps(self):                self.assertAlmostEqual(self.t.norm_op_eps, 1e-5)
    def test_norm_op_affine(self):             self.assertTrue(self.t.norm_op_affine)
    def test_dropout_p(self):                  self.assertAlmostEqual(self.t.dropout_p, 0.0)
    def test_dropout_in_localization(self):    self.assertFalse(self.t.dropout_in_localization)
    def test_nonlin_type(self):                self.assertEqual(self.t.nonlin_type, "LeakyReLU")
    def test_nonlin_negative_slope(self):      self.assertAlmostEqual(self.t.nonlin_negative_slope, 1e-2)
    def test_convolutional_pooling(self):      self.assertTrue(self.t.convolutional_pooling)
    def test_convolutional_upsampling(self):   self.assertTrue(self.t.convolutional_upsampling)
    def test_seg_output_use_bias(self):        self.assertFalse(self.t.seg_output_use_bias)
    def test_he_init_neg_slope(self):          self.assertAlmostEqual(self.t.he_init_neg_slope, 1e-2)

    # DA
    def test_aug_do_elastic(self):             self.assertFalse(self.t.aug_do_elastic)
    def test_aug_do_mirror_fast_default(self): self.assertFalse(self.t.aug_do_mirror)
    def test_aug_scale_range(self):            self.assertEqual(self.t.aug_scale_range, (0.7, 1.4))
    def test_aug_do_gamma(self):               self.assertTrue(self.t.aug_do_gamma)

    # Data loading
    def test_oversample_foreground_percent(self):
        self.assertAlmostEqual(self.t.oversample_foreground_percent, 0.33)
    def test_pin_memory(self):                 self.assertTrue(self.t.pin_memory)

    # Checkpointing / fast-batch
    def test_save_every(self):                 self.assertEqual(self.t.save_every, 50)
    def test_batch_size_forced_fast(self):     self.assertEqual(self.t.batch_size_forced_fast, 16)

    # HP_JSON_PATH must point to a real file
    def test_hp_json_path_exists(self):
        from nnunet.training.network_training.custom_trainers.nnUNetTrainerV2_configurable import (
            nnUNetTrainerV2_configurable as Cls,
        )
        self.assertTrue(os.path.isfile(Cls.HP_JSON_PATH))


# ══════════════════════════════════════════════════════════════════════════════
# 3. load_hyperparameters_from_json — one param per test
# ══════════════════════════════════════════════════════════════════════════════
class TestLoadHyperparametersFromJson(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _t(self, overrides: dict):
        return _make_trainer(self.tmpdir, extra_json=overrides)

    # --- Training loop -------------------------------------------------------
    def test_max_num_epochs(self):
        self.assertEqual(self._t({"max_num_epochs": 500}).max_num_epochs, 500)

    def test_num_batches_per_epoch(self):
        self.assertEqual(self._t({"num_batches_per_epoch": 112}).num_batches_per_epoch, 112)

    def test_num_val_batches_per_epoch(self):
        self.assertEqual(self._t({"num_val_batches_per_epoch": 5}).num_val_batches_per_epoch, 5)

    def test_patience(self):
        self.assertEqual(self._t({"patience": 100}).patience, 100)

    def test_val_eval_criterion_alpha(self):
        self.assertAlmostEqual(self._t({"val_eval_criterion_alpha": 0.95}).val_eval_criterion_alpha, 0.95)

    def test_train_loss_MA_alpha(self):
        self.assertAlmostEqual(self._t({"train_loss_MA_alpha": 0.80}).train_loss_MA_alpha, 0.80)

    def test_lr_threshold(self):
        self.assertAlmostEqual(self._t({"lr_threshold": 1e-7}).lr_threshold, 1e-7)

    # --- Optimizer -----------------------------------------------------------
    def test_initial_lr(self):
        self.assertAlmostEqual(self._t({"initial_lr_nnUNetTrainerV2": 0.007}).initial_lr, 0.007)

    def test_weight_decay(self):
        self.assertAlmostEqual(self._t({"weight_decay": 1e-4}).weight_decay, 1e-4)

    def test_sgd_momentum(self):
        self.assertAlmostEqual(self._t({"sgd_momentum": 0.95}).sgd_momentum, 0.95)

    def test_sgd_nesterov_false(self):
        self.assertFalse(self._t({"sgd_nesterov": False}).sgd_nesterov)

    def test_poly_lr_exponent(self):
        self.assertAlmostEqual(self._t({"poly_lr_exponent": 0.8}).poly_lr_exponent, 0.8)

    def test_grad_clip_norm(self):
        self.assertAlmostEqual(self._t({"grad_clip_norm": 5.0}).grad_clip_norm, 5.0)

    def test_momentum_rescue_threshold(self):
        self.assertAlmostEqual(
            self._t({"momentum_rescue_threshold": 0.90}).momentum_rescue_threshold, 0.90)

    # --- Loss function -------------------------------------------------------
    def test_weight_ce(self):
        self.assertAlmostEqual(self._t({"weight_ce": 1.5}).weight_ce, 1.5)

    def test_weight_dice(self):
        self.assertAlmostEqual(self._t({"weight_dice": 0.5}).weight_dice, 0.5)

    def test_dice_smooth(self):
        self.assertAlmostEqual(self._t({"dice_smooth": 1e-3}).dice_smooth, 1e-3)

    def test_dice_do_bg_true(self):
        self.assertTrue(self._t({"dice_do_bg": True}).dice_do_bg)

    def test_square_dice_true(self):
        self.assertTrue(self._t({"square_dice": True}).square_dice)

    def test_topk_k(self):
        self.assertEqual(self._t({"topk_k": 20}).topk_k, 20)

    def test_loss_function_topk(self):
        self.assertEqual(self._t({"loss_function": "DC_and_topk_loss"}).loss_function, "DC_and_topk_loss")

    # --- batch_dice: must be intercepted before super().__init__() -----------
    def test_batch_dice_false_via_json(self):
        self.assertFalse(self._t({"batch_dice": False}).batch_dice)

    def test_batch_dice_true_no_override(self):
        self.assertTrue(self._t({}).batch_dice)

    # --- Deep supervision ----------------------------------------------------
    def test_ds_lowest_levels_masked(self):
        self.assertEqual(self._t({"ds_lowest_levels_masked": 2}).ds_lowest_levels_masked, 2)

    def test_deep_supervision_disabled(self):
        self.assertFalse(self._t({"deep_supervision_enabled": False}).deep_supervision_enabled)

    # --- Network architecture ------------------------------------------------
    def test_norm_op_batchnorm(self):
        self.assertEqual(self._t({"norm_op": "BatchNorm"}).norm_op_type, "BatchNorm")

    def test_norm_op_eps(self):
        self.assertAlmostEqual(self._t({"norm_op_eps": 1e-3}).norm_op_eps, 1e-3)

    def test_dropout_p(self):
        self.assertAlmostEqual(self._t({"dropout_p": 0.2}).dropout_p, 0.2)

    def test_nonlin_relu(self):
        self.assertEqual(self._t({"nonlin": "ReLU"}).nonlin_type, "ReLU")

    def test_nonlin_negative_slope(self):
        self.assertAlmostEqual(
            self._t({"nonlin_negative_slope": 0.05}).nonlin_negative_slope, 0.05)

    def test_he_init_neg_slope(self):
        self.assertAlmostEqual(self._t({"he_init_neg_slope": 0.0}).he_init_neg_slope, 0.0)

    def test_convolutional_pooling_false(self):
        self.assertFalse(self._t({"convolutional_pooling": False}).convolutional_pooling)

    def test_convolutional_upsampling_false(self):
        self.assertFalse(self._t({"convolutional_upsampling": False}).convolutional_upsampling)

    def test_seg_output_use_bias_true(self):
        self.assertTrue(self._t({"seg_output_use_bias": True}).seg_output_use_bias)

    # --- Data augmentation ---------------------------------------------------
    def test_aug_do_elastic(self):
        self.assertTrue(self._t({"do_elastic": True}).aug_do_elastic)

    def test_aug_p_eldef(self):
        self.assertAlmostEqual(self._t({"p_eldef": 0.4}).aug_p_eldef, 0.4)

    def test_aug_scale_range(self):
        self.assertEqual(self._t({"scale_range": [0.5, 1.5]}).aug_scale_range, (0.5, 1.5))

    def test_aug_p_scale(self):
        self.assertAlmostEqual(self._t({"p_scale": 0.3}).aug_p_scale, 0.3)

    def test_aug_do_rotation_false(self):
        self.assertFalse(self._t({"do_rotation": False}).aug_do_rotation)

    def test_aug_p_rot(self):
        self.assertAlmostEqual(self._t({"p_rot": 0.3}).aug_p_rot, 0.3)

    def test_aug_gamma_range(self):
        self.assertEqual(self._t({"gamma_range": [0.5, 2.0]}).aug_gamma_range, (0.5, 2.0))

    def test_aug_gamma_retain_stats_false(self):
        self.assertFalse(self._t({"gamma_retain_stats": False}).aug_gamma_retain_stats)

    def test_aug_p_gamma(self):
        self.assertAlmostEqual(self._t({"p_gamma": 0.5}).aug_p_gamma, 0.5)

    def test_aug_do_mirror_enabled(self):
        self.assertTrue(self._t({"do_mirror": True}).aug_do_mirror)

    def test_aug_mirror_axes(self):
        self.assertEqual(self._t({"mirror_axes": [0, 1]}).aug_mirror_axes, (0, 1))

    def test_aug_do_additive_brightness(self):
        self.assertTrue(self._t({"do_additive_brightness": True}).aug_do_additive_brightness)

    def test_aug_additive_brightness_mu(self):
        self.assertAlmostEqual(
            self._t({"additive_brightness_mu": 0.05}).aug_additive_brightness_mu, 0.05)

    def test_aug_num_cached_per_thread(self):
        self.assertEqual(self._t({"num_cached_per_thread": 4}).aug_num_cached_per_thread, 4)

    def test_aug_border_mode_data(self):
        self.assertEqual(self._t({"border_mode_data": "nearest"}).aug_border_mode_data, "nearest")

    # --- moreDA stored-only params (no training effect, but should be stored) -
    def test_gaussian_noise_p_stored(self):
        self.assertAlmostEqual(self._t({"gaussian_noise_p": 0.3}).aug_gaussian_noise_p, 0.3)

    def test_contrast_p_stored(self):
        self.assertAlmostEqual(
            self._t({"contrast_augmentation_p": 0.2}).aug_contrast_p, 0.2)

    def test_simulate_lowres_p_stored(self):
        self.assertAlmostEqual(
            self._t({"simulate_lowres_p_per_sample": 0.4}).aug_lowres_p, 0.4)

    # --- Data loading --------------------------------------------------------
    def test_oversample_foreground_percent(self):
        self.assertAlmostEqual(
            self._t({"oversample_foreground_percent": 0.66}).oversample_foreground_percent, 0.66)

    def test_pin_memory_false(self):
        self.assertFalse(self._t({"pin_memory": False}).pin_memory)

    # --- Checkpointing -------------------------------------------------------
    def test_save_every(self):
        self.assertEqual(self._t({"save_every": 100}).save_every, 100)

    def test_save_latest_only_false(self):
        self.assertFalse(self._t({"save_latest_only": False}).save_latest_only)

    def test_save_best_checkpoint_false(self):
        self.assertFalse(self._t({"save_best_checkpoint": False}).save_best_checkpoint)

    def test_save_final_checkpoint_false(self):
        self.assertFalse(self._t({"save_final_checkpoint": False}).save_final_checkpoint)

    # --- Inference -----------------------------------------------------------
    def test_lr_scheduler_eps(self):
        self.assertAlmostEqual(self._t({"lr_scheduler_eps": 1e-4}).lr_scheduler_eps, 1e-4)

    def test_inference_pad_border_mode(self):
        self.assertEqual(
            self._t({"inference_pad_border_mode": "reflect"}).inference_pad_border_mode, "reflect")

    def test_inference_pad_constant_value(self):
        t = self._t({"inference_pad_constant_value": -1024.0})
        self.assertAlmostEqual(t.inference_pad_kwargs["constant_values"], -1024.0)

    # --- Fast-trainer batch size ---------------------------------------------
    def test_batch_size_forced_fast(self):
        self.assertEqual(self._t({"batch_size_forced_fast": 8}).batch_size_forced_fast, 8)

    # --- Multiple overrides simultaneously -----------------------------------
    def test_multiple_overrides(self):
        t = self._t({
            "oversample_foreground_percent": 0.66,
            "weight_ce":                    1.5,
            "batch_dice":                   False,
            "initial_lr_nnUNetTrainerV2":   0.007,
            "num_batches_per_epoch":        112,
            "grad_clip_norm":               5.0,
        })
        self.assertAlmostEqual(t.oversample_foreground_percent, 0.66)
        self.assertAlmostEqual(t.weight_ce, 1.5)
        self.assertFalse(t.batch_dice)
        self.assertAlmostEqual(t.initial_lr, 0.007)
        self.assertEqual(t.num_batches_per_epoch, 112)
        self.assertAlmostEqual(t.grad_clip_norm, 5.0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. _rebuild_loss — loss object type and configuration
# ══════════════════════════════════════════════════════════════════════════════
class TestRebuildLoss(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.t = _make_trainer(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_loss_is_DC_and_CE(self):
        from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
        self.assertIsInstance(self.t.loss, DC_and_CE_loss)

    def test_rebuild_preserves_DC_and_CE_type(self):
        from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
        self.t.weight_ce = 2.0
        self.t._rebuild_loss()
        self.assertIsInstance(self.t.loss, DC_and_CE_loss)

    def test_rebuild_switches_to_topk_loss(self):
        from nnunet.training.loss_functions.dice_loss import DC_and_topk_loss
        self.t.loss_function = "DC_and_topk_loss"
        self.t._rebuild_loss()
        self.assertIsInstance(self.t.loss, DC_and_topk_loss)

    def test_rebuild_applies_batch_dice_false(self):
        self.t.batch_dice = False
        self.t._rebuild_loss()
        self.assertFalse(self.t.loss.dc.batch_dice)

    def test_rebuild_before_initialize_no_ds_wrapper(self):
        from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
        self.assertFalse(self.t.was_initialized)
        self.t._rebuild_loss()
        self.assertNotIsInstance(self.t.loss, MultipleOutputLoss2)

    def test_rebuild_after_initialize_wraps_with_ds(self):
        from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
        # Simulate post-initialize state
        self.t.was_initialized = True
        self.t.ds_loss_weights = np.array([0.5333, 0.2667, 0.1333, 0.0667])
        self.t._rebuild_loss()
        self.assertIsInstance(self.t.loss, MultipleOutputLoss2)

    def test_weight_ce_reflected_in_loss(self):
        self.t.weight_ce = 3.0
        self.t._rebuild_loss()
        self.assertAlmostEqual(self.t.loss.weight_ce, 3.0)

    def test_weight_dice_reflected_in_loss(self):
        self.t.weight_dice = 0.5
        self.t._rebuild_loss()
        self.assertAlmostEqual(self.t.loss.weight_dice, 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# 5. setup_DA_params — writes into self.data_aug_params
# ══════════════════════════════════════════════════════════════════════════════
class TestSetupDAParams(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _call_setup(self, t):
        """Provide minimal attributes that setup_DA_params() reads."""
        t.threeD = True
        t.patch_size = np.array([128, 128, 128])
        t.net_num_pool_op_kernel_sizes = [[2, 2, 2]] * 5
        t.do_dummy_2D_aug = False
        t.use_mask_for_norm = {"CT": False}
        with patch(_SILENCE, return_value=None):
            t.setup_DA_params()

    def test_scale_range(self):
        t = _make_trainer(self.tmpdir, {"scale_range": [0.5, 1.6]})
        self._call_setup(t)
        self.assertEqual(t.data_aug_params["scale_range"], (0.5, 1.6))

    def test_do_elastic_true(self):
        t = _make_trainer(self.tmpdir, {"do_elastic": True})
        self._call_setup(t)
        self.assertTrue(t.data_aug_params["do_elastic"])

    def test_p_eldef(self):
        t = _make_trainer(self.tmpdir, {"p_eldef": 0.45})
        self._call_setup(t)
        self.assertAlmostEqual(t.data_aug_params["p_eldef"], 0.45)

    def test_do_mirror_true(self):
        t = _make_trainer(self.tmpdir, {"do_mirror": True})
        self._call_setup(t)
        self.assertTrue(t.data_aug_params["do_mirror"])

    def test_mirror_axes(self):
        t = _make_trainer(self.tmpdir, {"mirror_axes": [0, 2]})
        self._call_setup(t)
        self.assertEqual(t.data_aug_params["mirror_axes"], (0, 2))

    def test_gamma_range(self):
        t = _make_trainer(self.tmpdir, {"gamma_range": [0.5, 2.0]})
        self._call_setup(t)
        self.assertEqual(t.data_aug_params["gamma_range"], (0.5, 2.0))

    def test_num_cached_per_thread(self):
        t = _make_trainer(self.tmpdir, {"num_cached_per_thread": 4})
        self._call_setup(t)
        self.assertEqual(t.data_aug_params["num_cached_per_thread"], 4)

    def test_border_mode_data(self):
        t = _make_trainer(self.tmpdir, {"border_mode_data": "nearest"})
        self._call_setup(t)
        self.assertEqual(t.data_aug_params["border_mode_data"], "nearest")

    def test_additive_brightness_forwarded(self):
        t = _make_trainer(self.tmpdir, {
            "do_additive_brightness":    True,
            "additive_brightness_mu":    0.05,
            "additive_brightness_sigma": 0.2,
        })
        self._call_setup(t)
        self.assertTrue(t.data_aug_params["do_additive_brightness"])
        self.assertAlmostEqual(t.data_aug_params["additive_brightness_mu"], 0.05)
        self.assertAlmostEqual(t.data_aug_params["additive_brightness_sigma"], 0.2)

    def test_do_rotation_false(self):
        t = _make_trainer(self.tmpdir, {"do_rotation": False})
        self._call_setup(t)
        self.assertFalse(t.data_aug_params["do_rotation"])

    def test_p_rot(self):
        t = _make_trainer(self.tmpdir, {"p_rot": 0.4})
        self._call_setup(t)
        self.assertAlmostEqual(t.data_aug_params["p_rot"], 0.4)


# ══════════════════════════════════════════════════════════════════════════════
# 6. process_plans — batch size override
# ══════════════════════════════════════════════════════════════════════════════
class TestProcessPlans(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _plans_stub():
        return {
            "num_classes": 5,
            "num_modalities": 1,
            "all_classes": list(range(1, 6)),
            "use_mask_for_norm": {"CT": False},
            "keep_only_largest_region": None,
            "min_region_size_per_class": None,
            "transpose_forward": [0, 1, 2],
            "transpose_backward": [0, 1, 2],
            "modalities": {0: "CT"},
            "normalization_schemes": {0: "CT"},
            "dataset_properties": {
                "intensityproperties": None,
                "all_classes": list(range(1, 6)),
            },
            "plans_per_stage": {
                0: {
                    "batch_size": 2,
                    "num_pool_per_axis": [5, 5, 5],
                    "patch_size": np.array([128, 128, 128]),
                    "median_patient_size_in_voxels": np.array([300, 300, 300]),
                    "current_spacing": np.array([1.5, 1.5, 1.5]),
                    "original_spacing": np.array([1.5, 1.5, 1.5]),
                    "do_dummy_2D_data_aug": False,
                    "pool_op_kernel_sizes": [[2, 2, 2]] * 5,
                    "conv_kernel_sizes": [[3, 3, 3]] * 6,
                    "num_classes": 5,
                }
            },
            "base_num_features": 32,
            "conv_per_stage": 2,
        }

    def test_batch_size_forced_to_custom_value(self):
        t = _make_trainer(self.tmpdir, {"batch_size_forced_fast": 4})
        with patch(_SILENCE, return_value=None):
            t.process_plans(self._plans_stub())
        self.assertEqual(t.batch_size, 4)

    def test_batch_size_default_16(self):
        t = _make_trainer(self.tmpdir)
        with patch(_SILENCE, return_value=None):
            t.process_plans(self._plans_stub())
        self.assertEqual(t.batch_size, 16)


# ══════════════════════════════════════════════════════════════════════════════
# 7. initialize_optimizer_and_scheduler — SGD params
# ══════════════════════════════════════════════════════════════════════════════
class TestInitializeOptimizer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _opt(self, overrides):
        t = _make_trainer(self.tmpdir, overrides)
        t.network = torch.nn.Linear(4, 2)
        with patch(_SILENCE, return_value=None):
            t.initialize_optimizer_and_scheduler()
        return t

    def test_momentum_applied(self):
        t = self._opt({"sgd_momentum": 0.95})
        self.assertAlmostEqual(t.optimizer.param_groups[0]["momentum"], 0.95)

    def test_nesterov_false(self):
        t = self._opt({"sgd_nesterov": False})
        self.assertFalse(t.optimizer.param_groups[0]["nesterov"])

    def test_initial_lr(self):
        t = self._opt({"initial_lr_nnUNetTrainerV2": 0.003})
        self.assertAlmostEqual(t.optimizer.param_groups[0]["lr"], 0.003)

    def test_weight_decay(self):
        t = self._opt({"weight_decay": 1e-4})
        self.assertAlmostEqual(t.optimizer.param_groups[0]["weight_decay"], 1e-4)

    def test_lr_scheduler_is_none(self):
        t = self._opt({})
        self.assertIsNone(t.lr_scheduler)

    def test_optimizer_type_is_sgd(self):
        t = self._opt({})
        self.assertIsInstance(t.optimizer, torch.optim.SGD)


# ══════════════════════════════════════════════════════════════════════════════
# 8. maybe_update_lr — poly_lr_exponent
# ══════════════════════════════════════════════════════════════════════════════
class TestMaybeUpdateLr(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _ready(self, overrides):
        t = _make_trainer(self.tmpdir, overrides)
        t.network = torch.nn.Linear(4, 2)
        t.epoch = 0
        with patch(_SILENCE, return_value=None):
            t.initialize_optimizer_and_scheduler()
        return t

    def test_different_exponents_give_different_lr(self):
        from nnunet.training.learning_rate.poly_lr import poly_lr
        t1 = self._ready({"poly_lr_exponent": 0.9})
        t2 = self._ready({"poly_lr_exponent": 0.5})
        with patch(_SILENCE, return_value=None):
            t1.maybe_update_lr(epoch=500)
            t2.maybe_update_lr(epoch=500)
        lr1 = t1.optimizer.param_groups[0]["lr"]
        lr2 = t2.optimizer.param_groups[0]["lr"]
        self.assertNotAlmostEqual(lr1, lr2, places=6)
        # Verify exact formula
        self.assertAlmostEqual(lr1, poly_lr(500, 1000, 1e-2, 0.9), places=10)
        self.assertAlmostEqual(lr2, poly_lr(500, 1000, 1e-2, 0.5), places=10)

    def test_lr_monotonically_decreasing(self):
        t = self._ready({})
        lrs = []
        with patch(_SILENCE, return_value=None):
            for ep in [100, 300, 600, 900]:
                t.maybe_update_lr(epoch=ep)
                lrs.append(t.optimizer.param_groups[0]["lr"])
        self.assertEqual(lrs, sorted(lrs, reverse=True))


# ══════════════════════════════════════════════════════════════════════════════
# 9. run_iteration — grad_clip_norm is passed to clip_grad_norm_
# ══════════════════════════════════════════════════════════════════════════════
class TestRunIteration(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_ready(self, overrides=None):
        """Return a trainer wired with a tiny network, SGD, and a mock loss."""
        t = _make_trainer(self.tmpdir, overrides or {})
        t.network = torch.nn.Linear(4, 2)
        t.network.do_ds = False
        t.fp16 = False
        with patch(_SILENCE, return_value=None):
            t.initialize_optimizer_and_scheduler()
        # Replace loss with a trivial differentiable scalar — avoids any
        # shape dependency on the segmentation Dice/CE implementation.
        t.loss = lambda output, target: output.sum() * 0.0
        return t

    def test_custom_grad_clip_norm_used(self):
        t = self._make_ready({"grad_clip_norm": 3.0})
        batch = {"data": torch.randn(2, 4), "target": [torch.zeros(2, dtype=torch.long)]}
        captured = []
        orig = torch.nn.utils.clip_grad_norm_

        def mock_clip(params, max_norm):
            captured.append(max_norm)
            return orig(params, max_norm)

        with patch("torch.nn.utils.clip_grad_norm_", side_effect=mock_clip):
            t.run_iteration(iter([batch]), do_backprop=True, run_online_evaluation=False)

        self.assertEqual(len(captured), 1)
        self.assertAlmostEqual(captured[0], 3.0)

    def test_run_iteration_returns_scalar(self):
        t = self._make_ready()
        batch = {"data": torch.randn(2, 4), "target": [torch.zeros(2, dtype=torch.long)]}
        result = t.run_iteration(iter([batch]), do_backprop=False)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 0)  # scalar numpy array


# ══════════════════════════════════════════════════════════════════════════════
# 10. HP JSON file sanity checks
# ══════════════════════════════════════════════════════════════════════════════
class TestHpJsonFile(unittest.TestCase):

    def setUp(self):
        from nnunet.training.network_training.custom_trainers.nnUNetTrainerV2_configurable import (
            nnUNetTrainerV2_configurable as Cls,
        )
        self.path = Cls.HP_JSON_PATH
        with open(self.path) as fh:
            self.data = json.load(fh)

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(self.path))

    def test_is_valid_json_dict(self):
        self.assertIsInstance(self.data, dict)

    def test_minimum_param_count(self):
        self.assertGreater(len(self.data), 50)

    def test_all_entries_have_description(self):
        missing = [k for k, v in self.data.items() if "description" not in v]
        self.assertEqual(missing, [], f"Missing 'description': {missing}")

    def test_all_entries_have_default_value(self):
        missing = [k for k, v in self.data.items() if "default_value" not in v]
        self.assertEqual(missing, [], f"Missing 'default_value': {missing}")

    def test_no_active_values_in_shipped_json(self):
        found = [k for k, v in self.data.items() if "active_value" in v]
        self.assertEqual(found, [],
                         "hyperparameter_reference.json must ship without active_value entries")

    def test_known_params_present(self):
        required = [
            "oversample_foreground_percent", "batch_dice", "weight_ce",
            "initial_lr_nnUNetTrainerV2", "num_batches_per_epoch",
            "ds_lowest_levels_masked", "grad_clip_norm", "poly_lr_exponent",
        ]
        for p in required:
            self.assertIn(p, self.data, f"Expected param '{p}' missing from JSON")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)

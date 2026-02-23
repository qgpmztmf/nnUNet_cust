# nnUNet Complete Hyperparameter Reference

> Auto-generated from source code audit of `nnUNet_cust/`.
> Trainer hierarchy: `nnUNetTrainerV2_fast` / `nnUNetTrainerV2_fast_8000` → `nnUNetTrainerV2` → `nnUNetTrainer` → `NetworkTrainer`

---

## 1. Training Loop Control

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `max_num_epochs` | `network_trainer.py · NetworkTrainer.__init__:97` | `1000` | Yes | Maximum number of training epochs |
| `max_num_epochs` (V2) | `nnUNetTrainerV2.py · nnUNetTrainerV2.__init__:48` | `1000` | Yes | Overrides base; always runs full 1000 |
| `max_num_epochs` (fast) | `custom_trainers/nnUNetTrainerV2_fast.py:10` | `1000` | Yes | Same as V2 in fast trainer |
| `max_num_epochs` (fast_8000) | `custom_trainers/nnUNetTrainerV2_fast_8000.py:10` | `8000` | Yes | Extended training budget |
| `num_batches_per_epoch` | `network_trainer.py · NetworkTrainer.__init__:98` | `250` | Yes | Training iterations per epoch |
| `num_batches_per_epoch` (fast) | `custom_trainers/nnUNetTrainerV2_fast.py:11` | `32` | Yes | Reduced batches in fast trainer |
| `num_val_batches_per_epoch` | `network_trainer.py · NetworkTrainer.__init__:99` | `50` | Yes | Online validation iterations per epoch |
| `num_val_batches_per_epoch` (fast) | `custom_trainers/nnUNetTrainerV2_fast.py:12` | `1` | Yes | Minimal online val in fast trainer |
| `num_val_batches_per_epoch` (fast_8000) | `custom_trainers/nnUNetTrainerV2_fast_8000.py:12` | `4` | Yes | Slightly more val in 8000-epoch trainer |
| `also_val_in_tr_mode` | `network_trainer.py · NetworkTrainer.__init__:100` | `False` | Yes | Also compute val loss in train mode |

---

## 2. Early Stopping & Patience

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `patience` | `network_trainer.py · NetworkTrainer.__init__:91` | `50` | Yes | Epochs to wait before early stopping |
| `val_eval_criterion_alpha` | `network_trainer.py · NetworkTrainer.__init__:92` | `0.9` | Yes | EMA coefficient for validation criterion MA |
| `train_loss_MA_alpha` | `network_trainer.py · NetworkTrainer.__init__:95` | `0.93` | Yes | EMA coefficient for training loss MA |
| `train_loss_MA_eps` | `network_trainer.py · NetworkTrainer.__init__:96` | `5e-4` | Yes | Minimum improvement for patience counter reset |
| `lr_threshold` | `network_trainer.py · NetworkTrainer.__init__:101` | `1e-6` | Yes | LR floor: early stopping only triggers if LR ≤ this |
| `early_stopping_override` | `nnUNetTrainerV2.py · on_epoch_end:414` | N/A (hardcoded) | No | nnUNetTrainerV2 overrides early stopping — always runs to `max_num_epochs` |

---

## 3. Optimizer & Learning Rate

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `optimizer` (nnUNetTrainer) | `nnUNetTrainer.py · initialize_optimizer_and_scheduler:267` | `Adam` | Yes | Adam with amsgrad=True |
| `optimizer` (nnUNetTrainerV2) | `nnUNetTrainerV2.py · initialize_optimizer_and_scheduler:166` | `SGD` | Yes | SGD (replaces Adam in V2) |
| `initial_lr` (nnUNetTrainer) | `nnUNetTrainer.py · __init__:126` | `3e-4` | Yes | Starting LR for Adam |
| `initial_lr` (nnUNetTrainerV2) | `nnUNetTrainerV2.py · __init__:49` | `1e-2` | Yes | Starting LR for SGD (poly decay) |
| `weight_decay` | `nnUNetTrainer.py · __init__:127` | `3e-5` | Yes | L2 regularization (shared by V2) |
| `momentum` | `nnUNetTrainerV2.py · initialize_optimizer_and_scheduler:167` | `0.99` | Yes | SGD momentum |
| `nesterov` | `nnUNetTrainerV2.py · initialize_optimizer_and_scheduler:167` | `True` | Yes | Nesterov momentum enabled |
| `amsgrad` | `nnUNetTrainer.py · initialize_optimizer_and_scheduler:268` | `True` | Yes | AMSGrad variant of Adam |
| `poly_lr_exponent` | `nnUNetTrainerV2.py · maybe_update_lr:405` → `poly_lr.py:17` | `0.9` | Yes | Exponent in `(1 - epoch/max_epochs)^0.9` |
| `lr_scheduler` (nnUNetTrainer) | `nnUNetTrainer.py · initialize_optimizer_and_scheduler:269` | `ReduceLROnPlateau` | Yes | Scheduler for Adam: factor=0.2, patience=30 |
| `lr_scheduler` (nnUNetTrainerV2) | `nnUNetTrainerV2.py · initialize_optimizer_and_scheduler:168` | `None` | No | Poly-LR used instead; no scheduler object |
| `lr_scheduler_eps` | `nnUNetTrainer.py · __init__:124` | `1e-3` | Yes | Threshold for ReduceLROnPlateau |
| `lr_scheduler_patience` | `nnUNetTrainer.py · __init__:125` | `30` | Yes | Patience for ReduceLROnPlateau |
| `lr_scheduler_factor` | `nnUNetTrainer.py · initialize_optimizer_and_scheduler:270` | `0.2` | Yes | LR reduction factor for ReduceLROnPlateau |
| `momentum_rescue_at_epoch100` | `nnUNetTrainerV2.py · on_epoch_end:419-420` | `0.95` | No | If val Dice = 0 at epoch 100, momentum reduced to 0.95 and network re-initialized |

---

## 4. Gradient Clipping

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `grad_clip_norm` | `nnUNetTrainerV2.py · run_iteration:254,264` | `12` | Yes | `torch.nn.utils.clip_grad_norm_` max norm |

---

## 5. Mixed Precision / Reproducibility

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `fp16` | `network_trainer.py · NetworkTrainer.__init__:59` | `False` | Yes | Enable AMP (autocast + GradScaler) |
| `deterministic` | `network_trainer.py · NetworkTrainer.__init__:43,62-68` | `True` | Yes | Enables `cudnn.deterministic=True`, `benchmark=False`, seeds |
| `random_seed` | `network_trainer.py · NetworkTrainer.__init__:63-66` | `12345` | No (hardcoded) | NumPy, PyTorch, CUDA seed |
| `cudnn.benchmark` | `network_trainer.py · NetworkTrainer.__init__:68,71` | `False` (deterministic) / `True` (non-det) | Yes (via `deterministic` flag) | cuDNN auto-tuning |
| `pin_memory` | `nnUNetTrainerV2.py · __init__:53` | `True` | Yes | DataLoader pin_memory |

---

## 6. Checkpointing

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `save_every` | `network_trainer.py · NetworkTrainer.__init__:122` | `50` | Yes | Save intermediate checkpoint every N epochs |
| `save_latest_only` | `network_trainer.py · NetworkTrainer.__init__:123` | `True` | Yes | Only keep `model_latest.model`; no per-epoch files |
| `save_intermediate_checkpoints` | `network_trainer.py · NetworkTrainer.__init__:125` | `True` | Yes | Save `model_latest.model` periodically |
| `save_best_checkpoint` | `network_trainer.py · NetworkTrainer.__init__:126` | `True` | Yes | Save `model_best.model` on best val metric |
| `save_final_checkpoint` | `network_trainer.py · NetworkTrainer.__init__:127` | `True` | Yes | Save `model_final_checkpoint.model` at end |

---

## 7. Loss Functions

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `loss` | `nnUNetTrainer.py · __init__:108` | `DC_and_CE_loss` | Yes | Combined Dice + CrossEntropy loss |
| `batch_dice` | `nnUNetTrainer.py · __init__:107` | `True` | Yes | Compute Dice over whole batch vs. per-sample |
| `dice_smooth` | `nnUNetTrainer.py · __init__:108` | `1e-5` | Yes | Smoothing constant in SoftDiceLoss numerator/denominator |
| `dice_do_bg` | `nnUNetTrainer.py · __init__:108` | `False` | Yes | Exclude background class from Dice computation |
| `weight_dice` | `dice_loss.py · DC_and_CE_loss.__init__` | `1` | Yes | Weight for Dice component of combined loss |
| `weight_ce` | `dice_loss.py · DC_and_CE_loss.__init__` | `1` | Yes | Weight for CE component of combined loss |
| `square_dice` | `dice_loss.py · DC_and_CE_loss.__init__` | `False` | Yes | Use squared Dice (SoftDiceLossSquared) |
| `log_dice` | `dice_loss.py · DC_and_CE_loss.__init__` | `False` | Yes | Apply `-log` transform to Dice loss |
| `topk_k` | `TopK_loss.py · TopKLoss.__init__:24` | `10` | Yes | Top-K percent of hardest voxels for TopK CE loss |
| `gd_smooth_eps` | `dice_loss.py · GDL.forward` | `1e-6` | Yes | Epsilon in GDL volume denominator |

---

## 8. Deep Supervision

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `deep_supervision` | `nnUNetTrainerV2.py · initialize_network:158` | `True` | Yes | Enable multi-scale deep supervision outputs |
| `ds_loss_weight_base` | `nnUNetTrainerV2.py · initialize:81` | `1 / 2^i` (exponential) | Yes | Per-scale loss weight: level `i` → `1/2^i` |
| `ds_lowest_level_mask` | `nnUNetTrainerV2.py · initialize:84` | lowest 1 level masked | Partially | Masks out the lowest resolution level(s) |
| `deep_supervision_scales` | `nnUNetTrainerV2.py · setup_DA_params:348-349` | computed from `net_num_pool_op_kernel_sizes` | No (auto) | Output resolutions for DS segmentation targets |
| `do_ds_during_training` | `nnUNetTrainerV2.py · run_training:439` | `True` (set explicitly) | No | Network `do_ds` flag is `True` during train, `False` during val/inference |

---

## 9. Network Architecture (Generic_UNet)

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `base_num_features` (base) | `generic_UNet.py · Generic_UNet:171` / `experiment_planner_baseline_3DUNet.py:52` | `30` | Yes | Starting feature map count |
| `base_num_features` (v21) | `experiment_planner_baseline_3DUNet_v21.py:36` | `32` | Yes | Increased for AMP divisibility by 8 |
| `max_num_features_3D` | `generic_UNet.py · Generic_UNet:173` | `320` | Yes | Feature map cap for 3D networks |
| `max_num_features_2D` | `generic_UNet.py · Generic_UNet:179` | `480` | Yes | Feature map cap for 2D networks |
| `feat_map_mul_on_downscale` | `generic_UNet.py · Generic_UNet.__init__:185` | `2` | Yes | Feature multiplier per pooling stage |
| `num_conv_per_stage` | `generic_UNet.py · Generic_UNet.__init__:184` | `2` | Yes | Number of conv blocks per encoder/decoder stage |
| `conv_kernel_size` | `generic_UNet.py · Generic_UNet.__init__:44` (ConvDropoutNormNonlin default) | `3` | Yes (from plans) | Convolution kernel size (per-axis from plans) |
| `conv_stride` | `generic_UNet.py · Generic_UNet.__init__:214` | `1` | No | Conv stride (pooling done separately) |
| `conv_dilation` | `generic_UNet.py · Generic_UNet.__init__:214` | `1` | No | Dilation factor |
| `conv_bias` | `generic_UNet.py · Generic_UNet.__init__:214` | `True` | Yes | Bias term in conv layers |
| `pool_op_kernel_sizes` | `nnUNetTrainer.py · process_plans:353` | from plans (typically 2×2×2) | No (auto) | Per-stage pooling kernel sizes |
| `conv_kernel_sizes` | `nnUNetTrainer.py · process_plans:359` | from plans (typically 3×3×3) | No (auto) | Per-stage conv kernel sizes |
| `norm_op` | `nnUNetTrainerV2.py · initialize_network:143,148` | `InstanceNorm3d` / `InstanceNorm2d` | Yes | Normalization layer type |
| `norm_op_eps` | `nnUNetTrainerV2.py · initialize_network:150` | `1e-5` | Yes | InstanceNorm epsilon |
| `norm_op_affine` | `nnUNetTrainerV2.py · initialize_network:150` | `True` | Yes | InstanceNorm learnable scale/bias |
| `dropout_op` | `nnUNetTrainerV2.py · initialize_network:142,147` | `Dropout3d` / `Dropout2d` | Yes | Dropout layer type |
| `dropout_p` | `nnUNetTrainerV2.py · initialize_network:151` | `0` | Yes | Dropout probability (disabled by default) |
| `dropout_in_localization` | `nnUNetTrainerV2.py · initialize_network:158` | `False` | Yes | Apply dropout in decoder/localization path |
| `nonlin` | `nnUNetTrainerV2.py · initialize_network:152` | `nn.LeakyReLU` | Yes | Activation function |
| `nonlin_negative_slope` | `nnUNetTrainerV2.py · initialize_network:153` | `1e-2` | Yes | LeakyReLU negative slope |
| `convolutional_pooling` | `nnUNetTrainerV2.py · initialize_network:158` | `False` | Yes | Use strided conv instead of MaxPool |
| `convolutional_upsampling` | `nnUNetTrainerV2.py · initialize_network:159` | `False` | Yes | Use transposed conv instead of interpolation |
| `seg_output_use_bias` | `nnUNetTrainerV2.py · initialize_network:159` | `False` | Yes | Add bias to final segmentation conv |
| `upscale_logits` | `generic_UNet.py · Generic_UNet.__init__:188` | `False` | Yes | Upscale all DS outputs to full resolution |
| `upsample_mode` | `generic_UNet.py · Generic_UNet.__init__:230,238` | `bilinear` (2D) / `trilinear` (3D) | No (auto) | Interpolation mode for decoder upsampling |

---

## 10. Weight Initialization

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `weightInitializer` | `nnUNetTrainerV2.py · initialize_network:158` | `InitWeights_He(1e-2)` | Yes | He (Kaiming) normal initialization |
| `he_init_neg_slope` | `initialization.py · InitWeights_He.__init__` | `1e-2` | Yes | Negative slope for He init (matches LeakyReLU slope) |

---

## 11. Data Augmentation (3D)

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `do_elastic` | `default_data_augmentation.py:default_3D_augmentation_params` | `True` | Yes | Enable elastic deformation |
| `do_elastic` (V2) | `nnUNetTrainerV2.py · setup_DA_params:385` | `False` | Yes | nnUNetTrainerV2 disables elastic deformation |
| `elastic_deform_alpha` | `default_data_augmentation.py:43` | `(0., 900.)` | Yes | Elastic deformation alpha range |
| `elastic_deform_sigma` | `default_data_augmentation.py:44` | `(9., 13.)` | Yes | Elastic deformation sigma range |
| `p_eldef` | `default_data_augmentation.py:45` | `0.2` | Yes | Probability of elastic deformation per sample |
| `do_scaling` | `default_data_augmentation.py:47` | `True` | Yes | Enable random scaling |
| `scale_range` (default) | `default_data_augmentation.py:48` | `(0.85, 1.25)` | Yes | Scaling factor range |
| `scale_range` (V2) | `nnUNetTrainerV2.py · setup_DA_params:384` | `(0.7, 1.4)` | Yes | Wider scaling in nnUNetTrainerV2 |
| `independent_scale_factor_for_each_axis` | `default_data_augmentation.py:49` | `False` | Yes | Scale each axis independently |
| `p_scale` | `default_data_augmentation.py:52` | `0.2` | Yes | Probability of scaling per sample |
| `do_rotation` | `default_data_augmentation.py:54` | `True` | Yes | Enable random rotation |
| `rotation_x` (default 3D) | `default_data_augmentation.py:55` | `(-15°, +15°)` in radians | Yes | Rotation range around x-axis |
| `rotation_y` (default 3D) | `default_data_augmentation.py:56` | `(-15°, +15°)` in radians | Yes | Rotation range around y-axis |
| `rotation_z` (default 3D) | `default_data_augmentation.py:57` | `(-15°, +15°)` in radians | Yes | Rotation range around z-axis |
| `rotation_x` (V2 3D) | `nnUNetTrainerV2.py · setup_DA_params:353` | `(-30°, +30°)` in radians | Yes | Increased rotation in nnUNetTrainerV2 |
| `rotation_y` (V2 3D) | `nnUNetTrainerV2.py · setup_DA_params:354` | `(-30°, +30°)` in radians | Yes | Increased rotation in nnUNetTrainerV2 |
| `rotation_z` (V2 3D) | `nnUNetTrainerV2.py · setup_DA_params:355` | `(-30°, +30°)` in radians | Yes | Increased rotation in nnUNetTrainerV2 |
| `rotation_p_per_axis` | `default_data_augmentation.py:58` | `1` | Yes | Probability per axis of applying rotation |
| `p_rot` | `default_data_augmentation.py:59` | `0.2` | Yes | Probability of rotation per sample |
| `random_crop` | `default_data_augmentation.py:61` | `False` | Yes | Enable random cropping |
| `do_gamma` | `default_data_augmentation.py:64` | `True` | Yes | Enable gamma augmentation |
| `gamma_range` | `default_data_augmentation.py:65` | `(0.7, 1.5)` | Yes | Gamma value range |
| `gamma_retain_stats` | `default_data_augmentation.py:65` | `True` | Yes | Retain mean/std after gamma augmentation |
| `p_gamma` | `default_data_augmentation.py:67` | `0.3` | Yes | Probability of gamma per sample |
| `gamma_inverted_p` | `data_augmentation_moreDA.py:83` | `0.1` | Yes (hardcoded) | Probability of inverted gamma transform |
| `do_mirror` | `default_data_augmentation.py:69` | `True` | Yes | Enable mirroring (TTA and training) |
| `do_mirror` (fast) | `custom_trainers/nnUNetTrainerV2_fast.py:16` | `False` | Yes | Mirroring disabled in fast trainers |
| `mirror_axes` (3D) | `default_data_augmentation.py:70` | `(0, 1, 2)` | Yes | Axes to mirror |
| `mirror_axes` (2D) | `default_data_augmentation.py:89` | `(0, 1)` | Yes | Mirror axes for 2D |
| `dummy_2D` | `default_data_augmentation.py:72` | `False` | Yes | Treat 3D as 2D slices for anisotropic data |
| `border_mode_data` | `default_data_augmentation.py:74` | `"constant"` | Yes | Border fill mode for spatial transforms |
| `border_cval_data` | `data_augmentation_moreDA.py:69` | `0` | Yes (hardcoded) | Fill value for data border |
| `border_cval_seg` | `data_augmentation_moreDA.py:71` | `-1` (passed as `border_val_seg`) | Yes | Fill value for segmentation border |
| `order_data` | `data_augmentation_moreDA.py:45` (function param) | `3` | Yes | Spline interpolation order for data |
| `order_seg` | `data_augmentation_moreDA.py:45` (function param) | `1` | Yes | Spline interpolation order for segmentation |
| `do_additive_brightness` | `default_data_augmentation.py:76` | `False` | Yes | Enable additive brightness augmentation |
| `additive_brightness_p_per_sample` | `default_data_augmentation.py:77` | `0.15` | Yes | Probability of additive brightness per sample |
| `additive_brightness_p_per_channel` | `default_data_augmentation.py:78` | `0.5` | Yes | Probability per channel |
| `additive_brightness_mu` | `default_data_augmentation.py:79` | `0.0` | Yes | Mean of additive brightness distribution |
| `additive_brightness_sigma` | `default_data_augmentation.py:80` | `0.1` | Yes | Std of additive brightness distribution |

---

## 12. Data Augmentation — Intensity Transforms (moreDA only)

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `gaussian_noise_p` | `data_augmentation_moreDA.py:79` | `0.1` | Yes (hardcoded) | Probability of Gaussian noise per sample |
| `gaussian_blur_sigma` | `data_augmentation_moreDA.py:80` | `(0.5, 1.0)` | Yes (hardcoded) | Sigma range for Gaussian blur |
| `gaussian_blur_different_sigma_per_channel` | `data_augmentation_moreDA.py:80` | `True` | Yes (hardcoded) | Independent sigma per channel |
| `gaussian_blur_p_per_sample` | `data_augmentation_moreDA.py:80` | `0.2` | Yes (hardcoded) | Per-sample probability |
| `gaussian_blur_p_per_channel` | `data_augmentation_moreDA.py:80` | `0.5` | Yes (hardcoded) | Per-channel probability |
| `brightness_multiplicative_range` | `data_augmentation_moreDA.py:81` | `(0.75, 1.25)` | Yes (hardcoded) | Multiplier range for brightness |
| `brightness_multiplicative_p` | `data_augmentation_moreDA.py:81` | `0.15` | Yes (hardcoded) | Per-sample probability |
| `contrast_augmentation_p` | `data_augmentation_moreDA.py:84` | `0.15` | Yes (hardcoded) | Per-sample probability |
| `simulate_lowres_zoom_range` | `data_augmentation_moreDA.py:85` | `(0.5, 1.0)` | Yes (hardcoded) | Zoom range for low-resolution simulation |
| `simulate_lowres_p_per_channel` | `data_augmentation_moreDA.py:85` | `0.5` | Yes (hardcoded) | Per-channel probability |
| `simulate_lowres_p_per_sample` | `data_augmentation_moreDA.py:85` | `0.25` | Yes (hardcoded) | Per-sample probability |
| `simulate_lowres_order_downsample` | `data_augmentation_moreDA.py:86` | `0` | Yes (hardcoded) | Downsampling interpolation order |
| `simulate_lowres_order_upsample` | `data_augmentation_moreDA.py:86` | `3` | Yes (hardcoded) | Upsampling interpolation order |

---

## 13. Data Loading

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `num_threads` | `default_data_augmentation.py:82` | `12` or `$nnUNet_n_proc_DA` | Yes (env var) | Worker threads for MultiThreadedAugmenter |
| `num_cached_per_thread` | `default_data_augmentation.py:83` | `1` (default) / `2` (V2) | Yes | Batches cached per augmentation thread |
| `oversample_foreground_percent` | `nnUNetTrainer.py · __init__:129` | `0.33` | Yes | Fraction of batch samples that must contain foreground |
| `pad_mode` | `nnUNetTrainer.py · get_basic_generators:404` | `"constant"` | Yes | Padding mode for DataLoader |
| `memmap_mode` | `nnUNetTrainer.py · get_basic_generators:404` | `'r'` | Yes | Memory-mapped array read mode |
| `unpack_data` | `nnUNetTrainer.py · __init__:76` | `True` | Yes | Decompress NPZ to NPY before training |
| `use_nondetMultiThreadedAugmenter` | `nnUNetTrainerV2.py · initialize:112` | `False` | Yes | Use deterministic vs. non-deterministic augmenter |

---

## 14. Cross-Validation

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `n_splits` | `nnUNetTrainerV2.py · do_split:296` | `5` | Yes | Number of CV folds |
| `kfold_random_state` | `nnUNetTrainerV2.py · do_split:296` | `12345` | No (hardcoded) | Seed for KFold split reproducibility |
| `oof_split_ratio` | `nnUNetTrainerV2.py · do_split:323` | `0.8` | No (hardcoded) | Train fraction for out-of-range fold fallback |
| `oof_seed_offset` | `nnUNetTrainerV2.py · do_split:321` | `12345 + fold` | No | Seed offset for random 80/20 fallback |

---

## 15. Inference & Validation Configuration

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `do_mirroring` (val) | `nnUNetTrainerV2.py · validate:182` | `True` | Yes | Test-time augmentation via mirroring |
| `do_mirroring` (fast val) | `custom_trainers/nnUNetTrainerV2_fast.py · validate:22` | `False` (forced) | No | Fast trainers force TTA off |
| `use_sliding_window` | `nnUNetTrainer.py · validate:526` | `True` | Yes | Use sliding window for full-resolution inference |
| `step_size` | `nnUNetTrainer.py · validate:526` | `0.5` | Yes | Overlap step size (0.5 = 50% overlap) |
| `save_softmax` | `nnUNetTrainer.py · validate:526` | `True` | Yes | Save softmax predictions as NPZ |
| `use_gaussian` | `nnUNetTrainer.py · validate:526` | `True` | Yes | Gaussian weighting for sliding window |
| `all_in_gpu` | `nnUNetTrainer.py · validate:526` | `False` | Yes | Keep all predictions on GPU |
| `inference_pad_border_mode` | `nnUNetTrainer.py · __init__:118` | `"constant"` | Yes | Padding mode during inference |
| `inference_pad_kwargs` | `nnUNetTrainer.py · __init__:119` | `{'constant_values': 0}` | Yes | Padding fill value during inference |
| `mixed_precision` (inference) | `nnUNetTrainer.py · validate:602` | `self.fp16` | Yes | Use AMP during inference |
| `interpolation_order` | `nnUNetTrainer.py · validate:549` | `1` (default fallback) | Yes | Spline order for output resampling |
| `interpolation_order_z` | `nnUNetTrainer.py · validate:550` | `0` (default fallback) | Yes | Spline order for z-axis resampling |
| `run_postprocessing_on_folds` | `nnUNetTrainer.py · validate:529` | `True` | Yes | Run connected component postprocessing |

---

## 16. Experiment Planning (Plans File)

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `unet_base_num_features` | `experiment_planner_baseline_3DUNet.py:52` | `30` (base) / `32` (v21) | Yes | Base feature maps at highest resolution |
| `unet_max_num_filters` | `experiment_planner_baseline_3DUNet.py:53` | `320` | Yes | Maximum feature map count |
| `unet_max_numpool` | `experiment_planner_baseline_3DUNet.py:54` | `999` | Yes | Maximum number of pooling operations |
| `unet_min_batch_size` | `experiment_planner_baseline_3DUNet.py:55` | `2` | Yes | Minimum auto-planned batch size |
| `unet_featuremap_min_edge_length` | `experiment_planner_baseline_3DUNet.py:56` | `4` | Yes | Smallest allowed feature map edge (voxels) |
| `target_spacing_percentile` | `experiment_planner_baseline_3DUNet.py:58` | `50` (median) | Yes | Percentile for target resampling spacing |
| `anisotropy_threshold` | `experiment_planner_baseline_3DUNet.py:59` | `3` | Yes | Ratio above which axis is treated as anisotropic |
| `aniso_spacing_percentile` | `experiment_planner_baseline_3DUNet_v21.py:76` | `10` | Yes | Percentile for anisotropic axis spacing |
| `how_much_of_a_patient_must_network_see` | `experiment_planner_baseline_3DUNet.py:60` | `4` (1/4 patient) | Yes | Denominator: fraction of patient visible in patch |
| `batch_size_covers_max_percent` | `experiment_planner_baseline_3DUNet.py:61` | `0.05` | Yes | Batch cannot exceed 5% of dataset size |
| `conv_per_stage` (planner) | `experiment_planner_baseline_3DUNet.py:64` | `2` | Yes | Convolutions per encoder/decoder stage |
| `patch_size_init_mm` | `experiment_planner_baseline_3DUNet_v21.py:106` | `512` mm isotropic | Yes (hardcoded) | Starting patch size in mm before clipping |
| `vram_reference_3D` | `generic_UNet.py:182` | `520000000` (~496 MB) | No | Reference VRAM (bytes) for batch size computation |
| `vram_reference_2D` | `generic_UNet.py:181` | `19739648` (~18 MB) | No | Reference VRAM (bytes) for 2D batch size |

---

## 17. Preprocessing

| Parameter | Location | Default Value | Modifiable | Description |
|---|---|---|---|---|
| `default_num_threads` | `configuration.py:3` | `8` or `$nnUNet_def_n_proc` | Yes (env var) | Worker threads for preprocessing |
| `RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD` | `configuration.py:4` | `3` | Yes | Max spacing ratio before z-axis is resampled with NN separately |
| `resample_order_data` | `preprocessing.py` (GenericPreprocessor) | `3` (cubic) | Yes | Spline interpolation order for data resampling |
| `resample_order_seg_aniso` | `preprocessing.py` (GenericPreprocessor) | `1` | Yes | Seg interpolation on anisotropic axis |
| `resample_order_seg_iso` | `preprocessing.py` (GenericPreprocessor) | `0` (nearest) | Yes | Seg interpolation on isotropic axes |
| `ct_clip_percentile_low` | `preprocessing.py` (CT normalization) | `0.5` | Yes | Lower percentile for CT intensity clipping |
| `ct_clip_percentile_high` | `preprocessing.py` (CT normalization) | `99.5` | Yes | Upper percentile for CT intensity clipping |
| `foreground_sample_count` | `preprocessing.py` | `10000` | Yes (hardcoded) | Samples per class for foreground location caching |

---

## 18. Custom Trainer Summary (nnUNetTrainerV2_fast vs nnUNetTrainerV2_fast_8000)

| Parameter | `nnUNetTrainerV2_fast` | `nnUNetTrainerV2_fast_8000` |
|---|---|---|
| `max_num_epochs` | `1000` | `8000` |
| `num_batches_per_epoch` | `32` | `32` |
| `num_val_batches_per_epoch` | `1` | `4` |
| `batch_size` | `16` (forced) | `16` (forced) |
| `do_mirror` | `False` | `False` |
| Base class | `nnUNetTrainerV2` | `nnUNetTrainerV2` |

---

*Generated: 2026-02-22 | Codebase: `/scratch/work/tianmid1/nnUNet_cust/nnunet/`*

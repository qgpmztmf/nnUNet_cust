from nnunet.training.network_training.custom_trainers.nnUNetTrainerV2_fast import nnUNetTrainerV2_fast
from nnunet.training.dataloading.dataset_loading import DataLoader3D
from nnunet.training.dataloading.dataset_loading_classbalanced import DataLoader3D_ClassBalanced


class nnUNetTrainerV2_classBalanced(nnUNetTrainerV2_fast):
    """nnUNetTrainerV2_fast with class-balanced patch sampling.

    Sampling split per batch item:
        10%  class-balanced  — uniform over all foreground classes, pick a case
                               containing that class, anchor patch on it.
        23%  fg-oversample   — random case, anchor on any fg voxel (nnUNet default).
        67%  random          — random case, random patch location.

    Everything else (loss, augmentation, architecture) is identical to
    nnUNetTrainerV2_fast.
    """

    # Fractions must sum to 1.0
    CLASS_BALANCED_PERCENT = 0.10
    FG_OVERSAMPLE_PERCENT  = 0.23
    # Random = 1 - 0.10 - 0.23 = 0.67 (implicit)

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr = DataLoader3D_ClassBalanced(
                self.dataset_tr,
                self.basic_generator_patch_size,
                self.patch_size,
                self.batch_size,
                all_classes=self.classes,           # foreground class IDs (local)
                has_prev_stage=False,
                class_balanced_percent=self.CLASS_BALANCED_PERCENT,
                oversample_foreground_percent=self.FG_OVERSAMPLE_PERCENT,
                memmap_mode='r',
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
            )
            # Validation uses standard random sampling (no augmentation bias)
            dl_val = DataLoader3D(
                self.dataset_val,
                self.patch_size,
                self.patch_size,
                self.batch_size,
                False,
                oversample_foreground_percent=self.oversample_foreground_percent,
                memmap_mode='r',
                pad_mode="constant",
                pad_sides=self.pad_all_sides,
            )
        else:
            # 2D fallback: use parent implementation unchanged
            dl_tr, dl_val = super().get_basic_generators()

        return dl_tr, dl_val

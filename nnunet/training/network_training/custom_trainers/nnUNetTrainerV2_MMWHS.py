from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


class nnUNetTrainerV2_MMWHS(nnUNetTrainerV2):
    """
    Trainer tailored for MMWHS cardiac CT segmentation (Task620).

    Dataset: 16 training cases, 64×64×64 volumes, 5 cardiac structures.

    Key differences from the base nnUNetTrainerV2:
      - max_num_epochs = 2000  : small dataset benefits from more passes
      - batch_size = 2         : keeps gradient diversity across 16 small cases
      - mirror augmentation ON : cardiac structures benefit from left-right flips
      - fp16 = True            : mixed precision for faster iteration
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=True):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.max_num_epochs = 200
        self.num_batches_per_epoch = 20
        self.num_val_batches_per_epoch = 4
        self.initial_lr = 1e-4

    def process_plans(self, plans):
        super().process_plans(plans)
        self.batch_size = 16

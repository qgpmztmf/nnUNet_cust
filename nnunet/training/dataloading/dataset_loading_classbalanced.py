from collections import OrderedDict

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle, isfile

from nnunet.training.dataloading.dataset_loading import DataLoader3D


class DataLoader3D_ClassBalanced(DataLoader3D):
    """DataLoader3D with a three-way patch-sampling strategy:

        class_balanced_percent  (default 0.10)
            Pick one foreground class uniformly at random from *all* classes,
            then pick a case that actually contains it, then anchor the patch
            on a random voxel of that class.  This gives every class equal
            expected representation regardless of its volume.

        oversample_foreground_percent  (default 0.23)
            Same as the original nnUNet foreground oversampling: pick a random
            case, then anchor the patch on any random foreground voxel
            (classes weighted by their voxel count inside the case).

        remainder (default 0.67)
            Fully random patch: pick a random case, random bbox – identical to
            the original nnUNet behaviour for non-oversampled items.

    Parameters
    ----------
    all_classes : list[int]
        All foreground class IDs that the model is trained to predict
        (local label space, 1-based).  Typically ``plans['all_classes']``.
    class_balanced_percent : float
        Fraction of batch items sampled with class-balanced strategy.
    oversample_foreground_percent : float
        Fraction of batch items sampled with plain foreground oversampling.
    """

    def __init__(self, data, patch_size, final_patch_size, batch_size,
                 all_classes,
                 has_prev_stage=False,
                 class_balanced_percent=0.10,
                 oversample_foreground_percent=0.23,
                 memmap_mode="r", pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):

        super().__init__(data, patch_size, final_patch_size, batch_size,
                         has_prev_stage=has_prev_stage,
                         oversample_foreground_percent=oversample_foreground_percent,
                         memmap_mode=memmap_mode, pad_mode=pad_mode,
                         pad_kwargs_data=pad_kwargs_data, pad_sides=pad_sides)

        self.class_balanced_percent = class_balanced_percent
        self.all_classes = np.array(sorted(all_classes))  # e.g. [1..24]

        # Pre-build index: class_id -> list of case_ids that contain it.
        # Uses precomputed class_locations stored in each case's .pkl.
        self._class_to_cases = self._build_class_to_cases()
        # Keep only classes that actually appear in at least one training case.
        self._available_classes = np.array(
            [c for c in self.all_classes if c in self._class_to_cases]
        )
        missing = set(self.all_classes.tolist()) - set(self._available_classes.tolist())
        if missing:
            print(f"[ClassBalanced] WARNING: {len(missing)} class(es) absent from all "
                  f"training cases and will never be class-balanced sampled: {sorted(missing)}")
        print(f"[ClassBalanced] Index built: {len(self._available_classes)} classes "
              f"available for class-balanced sampling.")

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_class_to_cases(self):
        """Return dict {class_id: [case_id, ...]} from precomputed pkl files."""
        class_to_cases = {}
        for case_id in self.list_of_keys:
            if 'properties' in self._data[case_id]:
                props = self._data[case_id]['properties']
            else:
                props = load_pickle(self._data[case_id]['properties_file'])
            for cls, locs in props.get('class_locations', {}).items():
                if cls > 0 and len(locs) > 0:
                    class_to_cases.setdefault(int(cls), []).append(case_id)
        return class_to_cases

    # ------------------------------------------------------------------
    # Sampling mode decision
    # ------------------------------------------------------------------

    def _get_sampling_mode(self):
        """Stochastic 3-way decision per batch item."""
        r = np.random.rand()
        if r < self.class_balanced_percent:
            return 'class_balanced'
        elif r < self.class_balanced_percent + self.oversample_foreground_percent:
            return 'fg'
        else:
            return 'random'

    # ------------------------------------------------------------------
    # Class-balanced case / voxel selection
    # ------------------------------------------------------------------

    def _pick_class_balanced(self):
        """Return (case_id, selected_class, voxels_of_that_class).

        Picks a class uniformly at random from all_classes that appear in at
        least one training case, then picks a random case containing it.
        Falls back to plain fg-oversample if no suitable class exists.
        """
        if len(self._available_classes) == 0:
            return None, None, None  # caller will fall back to fg

        selected_class = int(np.random.choice(self._available_classes))
        case_id = np.random.choice(self._class_to_cases[selected_class])

        # Load voxel locations from cached properties
        if 'properties' in self._data[case_id]:
            props = self._data[case_id]['properties']
        else:
            props = load_pickle(self._data[case_id]['properties_file'])

        voxels = props['class_locations'].get(selected_class, None)
        if voxels is None or len(voxels) == 0:
            # Stale index — shouldn't normally happen
            return None, None, None

        return case_id, selected_class, voxels

    # ------------------------------------------------------------------
    # Main batch generation
    # ------------------------------------------------------------------

    def generate_train_batch(self):
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)
        case_properties = []
        selected_keys = []

        for j in range(self.batch_size):
            mode = self._get_sampling_mode()

            # ── Determine case_id, force_fg flag, and optionally pin voxels ──
            voxels_of_that_class = None  # used only in class_balanced mode

            if mode == 'class_balanced':
                case_id, selected_class, voxels_of_that_class = self._pick_class_balanced()
                if case_id is None:
                    # fallback: treat as fg-oversample
                    mode = 'fg'
                    case_id = np.random.choice(self.list_of_keys)
                force_fg = (mode == 'fg')
            elif mode == 'fg':
                case_id = np.random.choice(self.list_of_keys)
                force_fg = True
            else:  # random
                case_id = np.random.choice(self.list_of_keys)
                force_fg = False

            selected_keys.append(case_id)
            i = case_id

            # ── Load properties ──
            if 'properties' in self._data[i]:
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            # ── Load image+seg data ──
            npy_path = self._data[i]['data_file'][:-4] + ".npy"
            if isfile(npy_path):
                case_all_data = np.load(npy_path, self.memmap_mode, allow_pickle=True)
            else:
                case_all_data = np.load(self._data[i]['data_file'])['data']

            # ── Previous-stage cascade seg (if any) ──
            seg_from_previous_stage = None
            if self.has_prev_stage:
                ps_path = self._data[i]['seg_from_prev_stage_file'][:-4] + ".npy"
                if isfile(ps_path):
                    segs_from_previous_stage = np.load(ps_path, mmap_mode=self.memmap_mode)[None]
                else:
                    segs_from_previous_stage = np.load(
                        self._data[i]['seg_from_prev_stage_file'])['data'][None]
                seg_key = np.random.choice(segs_from_previous_stage.shape[0])
                seg_from_previous_stage = segs_from_previous_stage[seg_key:seg_key + 1]

            # ── Compute patch bounds ──
            need_to_pad = self.need_to_pad.copy()
            for d in range(3):
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = -need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = -need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
            lb_z = -need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

            # ── Choose patch anchor ──
            if mode == 'class_balanced' and voxels_of_that_class is not None:
                # Pin patch on a random voxel of the target class
                selected_voxel = voxels_of_that_class[
                    np.random.choice(len(voxels_of_that_class))]
                bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)

            elif force_fg:  # plain fg-oversample
                if 'class_locations' not in properties:
                    raise RuntimeError("Please rerun preprocessing with the newest nnU-Net!")

                foreground_classes = np.array([
                    c for c, locs in properties['class_locations'].items()
                    if len(locs) > 0 and c > 0
                ])
                if len(foreground_classes) == 0:
                    bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                    bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                    bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
                else:
                    sc = np.random.choice(foreground_classes)
                    voxels = properties['class_locations'][sc]
                    selected_voxel = voxels[np.random.choice(len(voxels))]
                    bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                    bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                    bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)

            else:  # random
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]
            bbox_z_ub = bbox_z_lb + self.patch_size[2]

            # ── Crop + pad ──
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            case_all_data = np.copy(
                case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                              valid_bbox_y_lb:valid_bbox_y_ub,
                              valid_bbox_z_lb:valid_bbox_z_ub])

            if seg_from_previous_stage is not None:
                seg_from_previous_stage = seg_from_previous_stage[
                    :, valid_bbox_x_lb:valid_bbox_x_ub,
                    valid_bbox_y_lb:valid_bbox_y_ub,
                    valid_bbox_z_lb:valid_bbox_z_ub]

            pad_x = (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0))
            pad_y = (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))
            pad_z = (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))

            data[j] = np.pad(case_all_data[:-1],
                             ((0, 0), pad_x, pad_y, pad_z),
                             self.pad_mode, **self.pad_kwargs_data)
            seg[j, 0] = np.pad(case_all_data[-1:],
                               ((0, 0), pad_x, pad_y, pad_z),
                               'constant', constant_values=-1)

            if seg_from_previous_stage is not None:
                seg[j, 1] = np.pad(seg_from_previous_stage,
                                   ((0, 0), pad_x, pad_y, pad_z),
                                   'constant', constant_values=0)

        return {'data': data, 'seg': seg,
                'properties': case_properties, 'keys': selected_keys}

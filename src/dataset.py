"""
src/dataset.py

Loads the S1S2 Landslide dataset from pre-split .h5 files.
Each .h5 file has shape (N, 1, 128, 128) per band key.

Structure discovered:
  - 355 samples in train, flat per-band layout
  - POST1_* keys for our target channels
  - None_MASK for segmentation labels
  - No NaN/Inf — data is clean
  - Images already 128x128 — no resize needed

Usage:
    from src.dataset import get_dataloaders
    loaders = get_dataloaders('data/s1s2_landslide', batch_size=16)
    for batch in loaders['train']:
        sar    = batch['sar']      # (B, 6, 128, 128)
        optical = batch['optical'] # (B, 11, 128, 128)
        mask   = batch['mask']     # (B, 1, 128, 128)
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


# ── Channel definitions ────────────────────────────────────────────────────────
# Exact key names as found in the .h5 files

SAR_KEYS = [
    'POST1_gVV',          # Gamma0 VV backscatter    — range: 0 to ~110
    'POST1_gVH',          # Gamma0 VH backscatter    — range: 0 to ~33
    'POST1_COHVV',        # Interferometric coherence — range: 0 to 0.99
    'POST1_ALPHA',        # PolSAR alpha angle        — range: 0.9 to 83
    'POST1_ENTROPY',      # PolSAR entropy            — range: 0.04 to 0.999
    'POST1_ANISOTROPY',   # PolSAR anisotropy         — range: 0.02 to 0.99
]

OPTICAL_KEYS = [
    'POST1_B02',    # Blue        — range: 1 to ~20640 (DN * 10000)
    'POST1_B03',    # Green       — range: 1 to ~19184
    'POST1_B04',    # Red         — range: 1 to ~18080
    'POST1_B05',    # Red-edge 1  — range: 0 to ~17558
    'POST1_B06',    # Red-edge 2  — range: 0 to ~16926
    'POST1_B07',    # Red-edge 3  — range: 0 to ~16614
    'POST1_B08',    # NIR broad   — range: 1 to ~16624
    'POST1_B8A',    # NIR narrow  — range: 1 to ~16540
    'POST1_B11',    # SWIR 1      — range: 31 to ~15151
    'POST1_B12',    # SWIR 2      — range: 10 to ~15080
    'POST1_NDVI',   # NDVI        — range: -0.99 to 0.999
]

MASK_KEY = 'None_MASK'   # Binary landslide mask — values: 0.0 or 1.0


# ── Normalisation functions ────────────────────────────────────────────────────

def normalise_sar(channel_name: str, data: np.ndarray) -> np.ndarray:
    """
    Normalise a single SAR channel to approximately [0, 1].

    Strategy per channel type:
      gVV / gVH        : log1p transform (compresses 0–110 range),
                         then min-max scale using fixed dataset statistics.
                         log1p(x) = log(1+x) so log1p(0) = 0 safely.

      COHVV            : already in [0, 1] — just clip to be safe.

      ALPHA            : min-max scale using observed range [0, 90].
                         Alpha is a polarimetric angle in degrees.

      ENTROPY          : already in [0, 1] — just clip.

      ANISOTROPY       : already in [0, 1] — just clip.
    """
    data = data.astype(np.float32)

    if 'gVV' in channel_name or 'gVH' in channel_name:
        # log1p compresses the heavy right tail (values up to 110)
        # After log1p: max becomes log(111) ≈ 4.71
        # We scale by fixed constant so train/val/test are consistent
        data = np.log1p(data)
        data = data / 5.0          # 5.0 > log1p(110) ≈ 4.71, so output in [0, ~0.94]
        data = np.clip(data, 0.0, 1.0)

    elif 'COHVV' in channel_name:
        data = np.clip(data, 0.0, 1.0)

    elif 'ALPHA' in channel_name:
        # Alpha angle physically ranges 0–90 degrees
        data = data / 90.0
        data = np.clip(data, 0.0, 1.0)

    elif 'ENTROPY' in channel_name:
        data = np.clip(data, 0.0, 1.0)

    elif 'ANISOTROPY' in channel_name:
        data = np.clip(data, 0.0, 1.0)

    return data


def normalise_optical(channel_name: str, data: np.ndarray) -> np.ndarray:
    """
    Normalise a single optical channel to [0, 1].

    Strategy per channel type:
      B02–B12  : Sentinel-2 L2A surface reflectance stored as DN*10000.
                 Divide by 10000 to get true reflectance in [0, 1].
                 Clip at 1.0 to handle rare saturated pixels (clouds etc.)

      NDVI     : Already in [-1, 1].
                 Shift to [0, 1] by (x + 1) / 2 so the model sees
                 all positive values — easier for normalised inputs.
    """
    data = data.astype(np.float32)

    if 'NDVI' in channel_name:
        data = (data + 1.0) / 2.0     # [-1,1] → [0,1]
        data = np.clip(data, 0.0, 1.0)

    else:
        # B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
        data = data / 10000.0
        data = np.clip(data, 0.0, 1.0)

    return data


# ── Dataset class ──────────────────────────────────────────────────────────────

class LandslideDataset(Dataset):
    """
    PyTorch Dataset for the S1S2 Landslide .h5 files.

    Each .h5 file stores ALL samples for a split together.
    Layout: h5_file[band_key][sample_index, 0, H, W]
    Shape per key: (N_samples, 1, 128, 128)

    This Dataset:
      - Opens the .h5 file once and keeps it open for fast random access
      - Returns per-sample dicts with normalised SAR, optical, and mask tensors
      - Shapes returned: sar (6,128,128), optical (11,128,128), mask (1,128,128)

    Args:
        h5_path  : full path to .h5 file (e.g. 'data/s1s2_landslide/train_n3_s1s2.h5')
        split    : string label for this split — used only for printed messages
    """

    def __init__(self, h5_path: str, split: str = 'train'):
        super().__init__()

        self.h5_path = h5_path
        self.split   = split

        # Validate file exists
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(
                f"Dataset file not found: {h5_path}\n"
                f"Make sure your data is in data/s1s2_landslide/"
            )

        # Open file once to read the number of samples
        # We keep self._file = None and open lazily in __getitem__
        # This is required for PyTorch multiprocessing (num_workers > 0)
        # h5py file handles cannot be shared across processes
        with h5py.File(self.h5_path, 'r') as f:
            # All keys have the same first dimension — use the first SAR key
            self.n_samples = f[SAR_KEYS[0]].shape[0]

        # h5py file handle — opened lazily per worker process
        self._file = None

        print(f"[Dataset] {split}: {self.n_samples} samples  |  "
              f"SAR channels: {len(SAR_KEYS)}  |  "
              f"Optical channels: {len(OPTICAL_KEYS)}  |  "
              f"File: {os.path.basename(h5_path)}")

    def _get_file(self) -> h5py.File:
        """
        Lazily open the h5 file in the current worker process.
        This is the correct pattern for h5py + PyTorch DataLoader
        with num_workers > 0.
        """
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        return self._file

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Load and return one sample.

        Args:
            idx : integer index in [0, n_samples)

        Returns:
            dict with keys:
              'sar'     : FloatTensor (6, 128, 128)   normalised SAR channels
              'optical' : FloatTensor (11, 128, 128)  normalised optical channels
              'mask'    : FloatTensor (1, 128, 128)   binary landslide mask
              'index'   : int                          sample index (for debugging)
        """
        f = self._get_file()

        # ── Load and normalise SAR channels ───────────────────────────────────
        sar_channels = []
        for key in SAR_KEYS:
            # h5 shape: (N, 1, 128, 128) → index [idx, 0] → (128, 128)
            raw = f[key][idx, 0, :, :]           # numpy (128, 128)
            norm = normalise_sar(key, raw)        # numpy (128, 128), float32
            sar_channels.append(norm)

        # Stack along channel dim → (6, 128, 128)
        sar = np.stack(sar_channels, axis=0)

        # ── Load and normalise optical channels ───────────────────────────────
        optical_channels = []
        for key in OPTICAL_KEYS:
            raw = f[key][idx, 0, :, :]
            norm = normalise_optical(key, raw)
            optical_channels.append(norm)

        # Stack → (11, 128, 128)
        optical = np.stack(optical_channels, axis=0)

        # ── Load mask ─────────────────────────────────────────────────────────
        # Shape: (N, 1, 128, 128) → index [idx] → (1, 128, 128)
        mask = f[MASK_KEY][idx, :, :, :]         # numpy (1, 128, 128)
        mask = mask.astype(np.float32)

        # ── Convert to PyTorch tensors ────────────────────────────────────────
        return {
            'sar'    : torch.from_numpy(sar),
            'optical': torch.from_numpy(optical),
            'mask'   : torch.from_numpy(mask),
            'index'  : idx,
        }

    def __del__(self):
        """Close h5 file handle when dataset is garbage collected."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloaders(
    data_root  : str,
    batch_size : int = 16,
    num_workers: int = 4,
    pin_memory : bool = True,
) -> dict:
    """
    Build train, val, and test DataLoaders from the three .h5 files.

    Args:
        data_root   : path to folder containing the three .h5 files
        batch_size  : number of samples per batch
        num_workers : parallel loading processes (0 = main process only)
        pin_memory  : faster GPU transfer when True (set False if RAM is tight)

    Returns:
        dict with keys 'train', 'val', 'test' — each a DataLoader

    File mapping:
        train_n3_s1s2.h5  →  355 samples  →  train loader (shuffled)
        val_n3_s1s2.h5    →  ? samples    →  val loader   (not shuffled)
        testspt_n3_s1s2.h5→  ? samples    →  test loader  (not shuffled)
    """
    splits = {
        'train' : os.path.join(data_root, 'train_n3_s1s2.h5'),
        'val'   : os.path.join(data_root, 'val_n3_s1s2.h5'),
        'test'  : os.path.join(data_root, 'testspt_n3_s1s2.h5'),
    }

    loaders = {}
    for split, path in splits.items():
        dataset = LandslideDataset(h5_path=path, split=split)
        loaders[split] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = (split == 'train'),   # shuffle only train
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = (split == 'train'),   # drop incomplete last batch in train
                                                # so batch size is always consistent
        )

    return loaders


# ── Shape verification test ────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Gate 3 check — run this directly to verify everything is correct:
        python src/dataset.py

    Expected output:
        [Dataset] train : 355 samples ...
        [Dataset] val   : X samples ...
        [Dataset] test  : X samples ...

        ── Single sample shapes ──
        sar     : torch.Size([6, 128, 128])   dtype: torch.float32
        optical : torch.Size([11, 128, 128])  dtype: torch.float32
        mask    : torch.Size([1, 128, 128])   dtype: torch.float32

        ── Batch shapes ──
        sar     : torch.Size([16, 6, 128, 128])
        optical : torch.Size([16, 11, 128, 128])
        mask    : torch.Size([16, 1, 128, 128])

        ── Value ranges after normalisation ──
        SAR    min: 0.0000  max: X.XXXX  (should be in [0, 1])
        Optical min: 0.0000  max: X.XXXX (should be in [0, 1])
        Mask   unique values: [0., 1.]   (binary)

        Gate 3 PASSED
    """
    import sys

    DATA_ROOT = 'data/s1s2_landslide'

    print("\n── Building DataLoaders ──────────────────────────────────────────")
    loaders = get_dataloaders(
        data_root   = DATA_ROOT,
        batch_size  = 16,
        num_workers = 0,    # 0 for this test — avoids multiprocessing issues on Windows
        pin_memory  = False,
    )

    print("\n── Single sample shapes ──────────────────────────────────────────")
    sample = loaders['train'].dataset[0]
    print(f"  sar     : {sample['sar'].shape}    dtype: {sample['sar'].dtype}")
    print(f"  optical : {sample['optical'].shape}  dtype: {sample['optical'].dtype}")
    print(f"  mask    : {sample['mask'].shape}    dtype: {sample['mask'].dtype}")

    # Assert correct shapes
    assert sample['sar'].shape     == (6, 128, 128),  f"SAR shape wrong: {sample['sar'].shape}"
    assert sample['optical'].shape == (11, 128, 128), f"Optical shape wrong: {sample['optical'].shape}"
    assert sample['mask'].shape    == (1, 128, 128),  f"Mask shape wrong: {sample['mask'].shape}"

    print("\n── Batch shapes ──────────────────────────────────────────────────")
    batch = next(iter(loaders['train']))
    print(f"  sar     : {batch['sar'].shape}")
    print(f"  optical : {batch['optical'].shape}")
    print(f"  mask    : {batch['mask'].shape}")

    assert batch['sar'].shape     == (16, 6, 128, 128)
    assert batch['optical'].shape == (16, 11, 128, 128)
    assert batch['mask'].shape    == (16, 1, 128, 128)

    print("\n── Value ranges after normalisation ──────────────────────────────")
    sar_min  = batch['sar'].min().item()
    sar_max  = batch['sar'].max().item()
    opt_min  = batch['optical'].min().item()
    opt_max  = batch['optical'].max().item()
    mask_vals = batch['mask'].unique()

    print(f"  SAR     min: {sar_min:.4f}  max: {sar_max:.4f}  (target: [0, 1])")
    print(f"  Optical min: {opt_min:.4f}  max: {opt_max:.4f}  (target: [0, 1])")
    print(f"  Mask    unique values: {mask_vals.tolist()}      (target: [0.0, 1.0])")

    assert sar_min >= -0.01,  "SAR has values below 0"
    assert sar_max <= 1.01,   "SAR has values above 1"
    assert opt_min >= -0.01,  "Optical has values below 0"
    assert opt_max <= 1.01,   "Optical has values above 1"
    assert not torch.isnan(batch['sar']).any(),     "SAR contains NaN"
    assert not torch.isnan(batch['optical']).any(), "Optical contains NaN"

    print("\n── Channel-level value check ─────────────────────────────────────")
    for i, key in enumerate(SAR_KEYS):
        ch = batch['sar'][:, i, :, :]
        print(f"  {key:25s}  min={ch.min():.3f}  max={ch.max():.3f}  mean={ch.mean():.3f}")

    print()
    for i, key in enumerate(OPTICAL_KEYS):
        ch = batch['optical'][:, i, :, :]
        print(f"  {key:25s}  min={ch.min():.3f}  max={ch.max():.3f}  mean={ch.mean():.3f}")

    print("\n── Speed test (iterating full train set) ─────────────────────────")
    import time
    t0 = time.time()
    for batch in loaders['train']:
        pass
    elapsed = time.time() - t0
    print(f"  Full train epoch: {elapsed:.1f}s  ({len(loaders['train'])} batches)")

    print("\n  Gate 3 PASSED — dataset.py is correct\n")
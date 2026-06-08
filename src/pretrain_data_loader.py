"""
Single-Trial Data Loader for Masked Self-Supervised Pretraining
================================================================

Unlike the contrastive data loader (which averages EEG repetitions down to one
trial per image), this loader keeps the individual repetitions. Self-supervised
pretraining does not use image labels or CLIP targets, so there is no reason to
throw away the extra trials — each repetition is more unlabeled EEG to learn
from. THINGS-EEG2 training has ~16,540 images x 4 reps per subject, so a single
subject already gives ~66k trials and all ten give several hundred thousand.

Returns batches of {'eeg': (B, 17, 250), 'subject_id': (B,)}.

Use --max-reps to cap repetitions per image when RAM is tight (each subject at
full reps is roughly 1 GB in float32).
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SingleTrialEEGDataset(Dataset):
    """All individual EEG trials from the chosen subjects, no averaging."""

    def __init__(self, eeg_data, subject_ids):
        self.eeg_data = eeg_data            # (N, 17, 250) float32
        self.subject_ids = subject_ids      # (N,) int64

        print(f"\nSingle-Trial Pretraining Dataset:")
        print(f"  Total trials : {len(self.eeg_data):,}")
        print(f"  EEG shape    : {self.eeg_data.shape}")
        print(f"  Subjects     : {sorted(np.unique(self.subject_ids).tolist())}")

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return {
            'eeg': torch.tensor(self.eeg_data[idx], dtype=torch.float32),
            'subject_id': torch.tensor(self.subject_ids[idx], dtype=torch.long),
        }


def create_pretrain_dataloader(
    preprocessed_path,
    subjects=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    batch_size=256,
    num_workers=0,
    max_reps=None,
):
    """
    Build a DataLoader of individual EEG trials for masked pretraining.

    Args:
        preprocessed_path : directory holding sub-XX/preprocessed_eeg_training.npy
        subjects          : subject IDs to include (1-10)
        batch_size        : batch size
        num_workers       : DataLoader workers
        max_reps          : if set, keep only the first `max_reps` repetitions
                            per image (lowers RAM); None keeps all repetitions.

    Returns:
        loader, num_subjects
    """
    print("=" * 70)
    print("CREATING SINGLE-TRIAL PRETRAINING DATALOADER")
    print("=" * 70)
    print(f"Subjects : {list(subjects)}")
    print(f"max_reps : {max_reps if max_reps else 'all'}")

    all_eeg = []
    all_sids = []

    for local_idx, subject_id in enumerate(subjects):
        subject_dir = os.path.join(preprocessed_path, f'sub-{subject_id:02d}')
        path = os.path.join(subject_dir, 'preprocessed_eeg_training.npy')

        with open(path, 'rb') as f:
            train_dict = pickle.load(f)
        eeg = train_dict['preprocessed_eeg_data']      # (n_img, n_rep, 17, 250)

        if eeg.ndim != 4:
            raise ValueError(
                f"Expected 4D (img, rep, ch, time) for sub-{subject_id:02d}, "
                f"got shape {eeg.shape}"
            )

        n_img, n_rep = eeg.shape[0], eeg.shape[1]
        if max_reps is not None:
            n_rep = min(n_rep, max_reps)
            eeg = eeg[:, :n_rep]

        # Flatten image x rep -> trials
        eeg = eeg.reshape(n_img * n_rep, eeg.shape[2], eeg.shape[3])
        eeg = eeg.astype(np.float32)

        sids = np.full(len(eeg), local_idx, dtype=np.int64)

        all_eeg.append(eeg)
        all_sids.append(sids)
        print(f"  sub-{subject_id:02d}: {n_img} imgs x {n_rep} reps "
              f"= {len(eeg):,} trials")

    all_eeg = np.concatenate(all_eeg, axis=0)
    all_sids = np.concatenate(all_sids, axis=0)

    dataset = SingleTrialEEGDataset(all_eeg, all_sids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"\nDataloader ready: {len(loader)} batches/epoch "
          f"@ batch_size={batch_size}")
    return loader, len(subjects)

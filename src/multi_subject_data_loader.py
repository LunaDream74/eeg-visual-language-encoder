"""
Multi-Subject Data Loader for THINGS-EEG2

Loads EEG data from multiple subjects simultaneously for joint training.
Handles subject-specific normalization and subject ID tracking.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class MultiSubjectDataset(Dataset):
    """Dataset that combines multiple subjects"""
    
    def __init__(self, all_eeg_data, all_image_ids, all_subject_ids, 
                 clip_embeddings, is_test=False):
        """
        Args:
            all_eeg_data: List of EEG arrays from each subject
            all_image_ids: List of image ID arrays from each subject
            all_subject_ids: List of subject IDs for each sample
            clip_embeddings: Shared CLIP embeddings (16740, 768)
            is_test: Whether this is test data
        """
        # Concatenate all subjects
        self.eeg_data = np.concatenate(all_eeg_data, axis=0)
        self.image_ids = np.concatenate(all_image_ids, axis=0)
        self.subject_ids = np.concatenate(all_subject_ids, axis=0)
        self.clip_embeddings = clip_embeddings
        self.is_test = is_test
        
        # Get unique subjects for logging
        unique_subjects = np.unique(self.subject_ids)
        
        print(f"\n{'Test' if is_test else 'Train'} Multi-Subject Dataset:")
        print(f"  Total samples: {len(self.eeg_data)}")
        print(f"  Subjects: {sorted(unique_subjects.tolist())}")
        print(f"  EEG shape: {self.eeg_data.shape}")
        print(f"  Image ID range: {self.image_ids.min()}-{self.image_ids.max()}")
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        image_id = self.image_ids[idx]
        subject_id = self.subject_ids[idx]
        clip_emb = self.clip_embeddings[image_id]
        
        # Normalize CLIP
        clip_emb = clip_emb / (np.linalg.norm(clip_emb) + 1e-8)
        
        return {
            'eeg': torch.tensor(eeg, dtype=torch.float32),
            'clip_emb': torch.tensor(clip_emb, dtype=torch.float32),
            'subject_id': torch.tensor(subject_id, dtype=torch.long),
            'image_id': image_id
        }


def create_multi_subject_dataloaders(
    preprocessed_path,
    clip_embeddings_path,
    subjects=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    batch_size=512,
    num_workers=0,
    val_split=0.1
):
    """
    Create multi-subject dataloaders
    
    Args:
        preprocessed_path: Path to preprocessed data directory
        clip_embeddings_path: Path to CLIP embeddings
        subjects: List of subject IDs to include (1-10)
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Validation split ratio
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    print("="*70)
    print("CREATING MULTI-SUBJECT DATALOADERS")
    print("="*70)
    print(f"Subjects: {subjects}")
    print(f"Batch size: {batch_size}")
    
    # Load CLIP embeddings (shared across all subjects)
    clip_embeddings = np.load(clip_embeddings_path)
    print(f"\nCLIP embeddings: {clip_embeddings.shape}")
    
    # Storage for all subjects
    all_train_eeg = []
    all_train_ids = []
    all_train_subjects = []
    
    all_test_eeg = []
    all_test_ids = []
    all_test_subjects = []
    
    # Load each subject
    for local_idx, subject_id in enumerate(subjects):
        subject_dir = os.path.join(preprocessed_path, f'sub-{subject_id:02d}')
        
        print(f"\nLoading Subject {subject_id} (local index {local_idx})...")
        
        # Load training data
        with open(os.path.join(subject_dir, 'preprocessed_eeg_training.npy'), 'rb') as f:
            train_dict = pickle.load(f)
        train_eeg = train_dict['preprocessed_eeg_data']
        
        # Load test data
        with open(os.path.join(subject_dir, 'preprocessed_eeg_test.npy'), 'rb') as f:
            test_dict = pickle.load(f)
        test_eeg = test_dict['preprocessed_eeg_data']
        
        print(f"  Training: {train_eeg.shape}")
        print(f"  Test: {test_eeg.shape}")
        
        # Average repetitions
        train_eeg_avg = train_eeg.mean(axis=1)  # (16540, 17, 250)
        test_eeg_avg = test_eeg.mean(axis=1)    # (200, 17, 250)
        
        # Image IDs (same for all subjects)
        train_image_ids = np.arange(0, 16540)
        test_image_ids = np.arange(16540, 16740)
        
        # Subject IDs — use position in list (0-indexed), NOT subject_number-1.
        # This means subject 2 alone → id=0, subjects [3,7] → ids 0,1, etc.
        # The model embedding table has size=len(subjects), so ids must be 0..N-1.
        train_subject_ids = np.full(len(train_eeg_avg), local_idx, dtype=np.int64)
        test_subject_ids  = np.full(len(test_eeg_avg),  local_idx, dtype=np.int64)
        
        # Accumulate
        all_train_eeg.append(train_eeg_avg)
        all_train_ids.append(train_image_ids)
        all_train_subjects.append(train_subject_ids)
        
        all_test_eeg.append(test_eeg_avg)
        all_test_ids.append(test_image_ids)
        all_test_subjects.append(test_subject_ids)
    
    # Concatenate all subjects
    all_train_eeg = np.concatenate(all_train_eeg, axis=0)
    all_train_ids = np.concatenate(all_train_ids, axis=0)
    all_train_subjects = np.concatenate(all_train_subjects, axis=0)
    
    all_test_eeg = np.concatenate(all_test_eeg, axis=0)
    all_test_ids = np.concatenate(all_test_ids, axis=0)
    all_test_subjects = np.concatenate(all_test_subjects, axis=0)
    
    print(f"\nCombined Data:")
    print(f"  Training: {all_train_eeg.shape} from {len(subjects)} subjects")
    print(f"  Test: {all_test_eeg.shape} from {len(subjects)} subjects")
    
    # Train/val split (stratified by subject if possible)
    num_train = len(all_train_eeg)
    num_val = int(num_train * val_split)
    
    # Shuffle
    indices = np.random.permutation(num_train)
    train_indices = indices[:num_train - num_val]
    val_indices = indices[num_train - num_val:]
    
    # Create datasets
    train_dataset = MultiSubjectDataset(
        [all_train_eeg[train_indices]],
        [all_train_ids[train_indices]],
        [all_train_subjects[train_indices]],
        clip_embeddings,
        is_test=False
    )
    
    val_dataset = MultiSubjectDataset(
        [all_train_eeg[val_indices]],
        [all_train_ids[val_indices]],
        [all_train_subjects[val_indices]],
        clip_embeddings,
        is_test=False
    )
    
    test_dataset = MultiSubjectDataset(
        [all_test_eeg],
        [all_test_ids],
        [all_test_subjects],
        clip_embeddings,
        is_test=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataloaders Created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print("="*70)
    
    return train_loader, val_loader, test_loader, len(subjects)


if __name__ == "__main__":
    # Test with 3 subjects first
    train_loader, val_loader, test_loader, num_subjects = create_multi_subject_dataloaders(
        preprocessed_path='./preprocessed_data_250Hz',
        clip_embeddings_path='THINGS_clip_embeddings/clip_embeddings_image_level.npy',
        subjects=[1, 2, 3],  # Start small!
        batch_size=512
    )
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nTest Batch:")
    print(f"  EEG: {batch['eeg'].shape}")
    print(f"  CLIP: {batch['clip_emb'].shape}")
    print(f"  Subject IDs: {batch['subject_id'].unique()}")
    print(f"  Image IDs: {batch['image_id'].min()}-{batch['image_id'].max()}")
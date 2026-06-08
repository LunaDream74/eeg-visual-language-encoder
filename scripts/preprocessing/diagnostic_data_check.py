"""
CRITICAL DIAGNOSTIC: Check Data Alignment & Preprocessing Quality

This will verify:
1. Preprocessed data format
2. CLIP embeddings format
3. Data-embedding alignment
4. Preprocessing quality
"""

import numpy as np
import torch
from old_mostly_infonce.data_loader_new_preprocessing import THINGSPreprocessedDataset

# =============================================================================
# CONFIGURATION
# =============================================================================

PREPROCESSED_PATH = "preprocessed_data"
CLIP_EMBEDDINGS_PATH = "THINGS_clip_embeddings/things_clip_embeddings.npy"
SUBJECT = 2

print("="*70)
print("DIAGNOSTIC: DATA ALIGNMENT & QUALITY CHECK")
print("="*70)

# =============================================================================
# 1. Check Preprocessed EEG Data
# =============================================================================

print("\n1. CHECKING PREPROCESSED EEG DATA")
print("-"*70)

import os
train_file = os.path.join(PREPROCESSED_PATH, f'sub-{SUBJECT:02d}', 'preprocessed_eeg_training.npy')
test_file = os.path.join(PREPROCESSED_PATH, f'sub-{SUBJECT:02d}', 'preprocessed_eeg_test.npy')

train_data = np.load(train_file, allow_pickle=True).item()
test_data = np.load(test_file, allow_pickle=True).item()

print(f"\nTraining data:")
print(f"  Shape: {train_data['preprocessed_eeg_data'].shape}")
print(f"  Expected: (1654, n_reps, 17, 250)")
print(f"  Channels: {len(train_data['ch_names'])}")
print(f"  Expected channels: 17")
print(f"  Timepoints: {len(train_data['times'])}")
print(f"  Expected timepoints: 250")
print(f"  Time range: {train_data['times'][0]:.3f}s to {train_data['times'][-1]:.3f}s")
print(f"  Expected: -0.2s to 0.8s")

# Check for issues
issues = []
if train_data['preprocessed_eeg_data'].shape[0] != 1654:
    issues.append(f" Wrong number of images: {train_data['preprocessed_eeg_data'].shape[0]} (expected 1654)")
if train_data['preprocessed_eeg_data'].shape[2] != 17:
    issues.append(f" Wrong number of channels: {train_data['preprocessed_eeg_data'].shape[2]} (expected 17)")
if train_data['preprocessed_eeg_data'].shape[3] != 250:
    issues.append(f" Wrong number of timepoints: {train_data['preprocessed_eeg_data'].shape[3]} (expected 250)")

time_window = train_data['times'][-1] - train_data['times'][0]
if abs(time_window - 1.0) > 0.05:
    issues.append(f" Wrong time window: {time_window:.3f}s (expected 1.0s)")

if issues:
    print("\n  ISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n Preprocessing format looks correct")

# Check data statistics
train_eeg = train_data['preprocessed_eeg_data']
print(f"\nData statistics:")
print(f"  Mean: {train_eeg.mean():.4f} (should be close to 0 if whitened)")
print(f"  Std: {train_eeg.std():.4f} (should be close to 1 if whitened)")
print(f"  Min: {train_eeg.min():.4f}")
print(f"  Max: {train_eeg.max():.4f}")

if abs(train_eeg.mean()) > 0.1:
    print(f"    Mean not close to 0 - whitening might have failed")
if abs(train_eeg.std() - 1.0) > 0.3:
    print(f"    Std not close to 1 - whitening might have failed")

# =============================================================================
# 2. Check CLIP Embeddings
# =============================================================================

print("\n2. CHECKING CLIP EMBEDDINGS")
print("-"*70)

clip_embeddings = np.load(CLIP_EMBEDDINGS_PATH)
print(f"\nCLIP embeddings shape: {clip_embeddings.shape}")
print(f"Expected: (1854, 512)")

if clip_embeddings.shape != (1854, 512):
    print(f" WRONG SHAPE! Expected (1854, 512)")
    print(f"   This is likely the problem!")

# Check if normalized
norms = np.linalg.norm(clip_embeddings, axis=1)
print(f"\nEmbedding norms:")
print(f"  Mean: {norms.mean():.4f}")
print(f"  Min: {norms.min():.4f}")
print(f"  Max: {norms.max():.4f}")

if abs(norms.mean() - 1.0) < 0.01:
    print(f"   Embeddings are normalized (unit vectors)")
else:
    print(f"    Embeddings might not be normalized")

# =============================================================================
# 3. Check Data Loader Indexing
# =============================================================================

print("\n3. CHECKING DATA LOADER INDEXING")
print("-"*70)

# Create datasets
train_dataset = THINGSPreprocessedDataset(
    preprocessed_path=PREPROCESSED_PATH,
    clip_embeddings_path=CLIP_EMBEDDINGS_PATH,
    subject_id=SUBJECT,
    split='train',
    augment=False,
    average_repetitions=True
)

test_dataset = THINGSPreprocessedDataset(
    preprocessed_path=PREPROCESSED_PATH,
    clip_embeddings_path=CLIP_EMBEDDINGS_PATH,
    subject_id=SUBJECT,
    split='test',
    augment=False,
    average_repetitions=True
)

# Check a few samples
print(f"\nTraining dataset:")
print(f"  Length: {len(train_dataset)}")
print(f"  Expected: 1654")

sample_train = train_dataset[0]
print(f"\n  Sample 0:")
print(f"    EEG shape: {sample_train['eeg'].shape}")
print(f"    CLIP shape: {sample_train['clip_emb'].shape}")
print(f"    Image ID: {sample_train['image_id']}")
print(f"    Expected image ID: 0")

if sample_train['image_id'] != 0:
    print(f"     WRONG! Image ID should be 0")

print(f"\nTest dataset:")
print(f"  Length: {len(test_dataset)}")
print(f"  Expected: 200")

sample_test = test_dataset[0]
print(f"\n  Sample 0:")
print(f"    EEG shape: {sample_test['eeg'].shape}")
print(f"    CLIP shape: {sample_test['clip_emb'].shape}")
print(f"    Image ID: {sample_test['image_id']}")
print(f"    Expected image ID: 1654")

if sample_test['image_id'] != 1654:
    print(f"     WRONG! Image ID should be 1654 (first test image)")

# =============================================================================
# 4. Check if EEG-CLIP alignment makes sense
# =============================================================================

print("\n4. CHECKING EEG-CLIP ALIGNMENT")
print("-"*70)

# Get a few samples
n_check = 5
print(f"\nChecking {n_check} training samples...")

for i in range(n_check):
    sample = train_dataset[i]
    eeg = sample['eeg']
    clip_emb = sample['clip_emb']
    img_id = sample['image_id']
    
    # Check if CLIP embedding matches what's in the file
    expected_clip = clip_embeddings[img_id]
    matches = np.allclose(clip_emb.numpy(), expected_clip, atol=1e-5)
    
    print(f"\nSample {i}:")
    print(f"  Image ID: {img_id}")
    print(f"  EEG mean: {eeg.mean():.4f}, std: {eeg.std():.4f}")
    print(f"  CLIP norm: {torch.norm(clip_emb).item():.4f}")
    print(f"  CLIP matches file: {matches}")
    
    if not matches:
        print(f"   CLIP embedding doesn't match file!")

# =============================================================================
# 5. Sanity check: Can we distinguish ANYTHING?
# =============================================================================

print("\n5. SANITY CHECK: EEG SIGNAL QUALITY")
print("-"*70)

# Get 100 samples
n_samples = min(100, len(train_dataset))
eeg_samples = []
for i in range(n_samples):
    eeg_samples.append(train_dataset[i]['eeg'].numpy())

eeg_samples = np.array(eeg_samples)  # (100, 17, 250)

# Compute pairwise correlations
print(f"\nComparing {n_samples} EEG samples...")
from scipy.spatial.distance import pdist
from scipy.stats import describe

# Flatten each sample
eeg_flat = eeg_samples.reshape(n_samples, -1)

# Compute pairwise correlations
correlations = []
for i in range(min(50, n_samples)):
    for j in range(i+1, min(50, n_samples)):
        corr = np.corrcoef(eeg_flat[i], eeg_flat[j])[0, 1]
        correlations.append(corr)

stats = describe(correlations)
print(f"\nPairwise correlation statistics:")
print(f"  Mean: {stats.mean:.4f}")
print(f"  Min: {stats.minmax[0]:.4f}")
print(f"  Max: {stats.minmax[1]:.4f}")

if stats.mean > 0.5:
    print(f"    WARNING: Very high average correlation!")
    print(f"     EEG samples might be too similar (not enough diversity)")
    print(f"     This suggests preprocessing might have gone wrong")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

print("\nKEY FINDINGS:")
if not issues:
    print(" Preprocessing format correct")
else:
    print(" Preprocessing issues found:")
    for issue in issues:
        print(f"  {issue}")

print(f"\nNext steps:")
print(f"1. If preprocessing format is wrong → Re-preprocess data")
print(f"2. If CLIP embeddings are wrong → Check CLIP embedding file")
print(f"3. If image IDs are wrong → Data loader has indexing bug")
print(f"4. If correlations are too high → Preprocessing removed signal")

print("\n" + "="*70)

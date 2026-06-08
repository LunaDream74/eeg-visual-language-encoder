"""
Test Data Diagnostic
Check why test accuracy (1.5%) is so much lower than validation (12.15%)
"""

import numpy as np
import torch
import sys

print("="*70)
print("TEST DATA DIAGNOSTIC")
print("="*70)

# =============================================================================
# 1. Check Test EEG Data Structure
# =============================================================================

print("\n1. Checking TEST EEG Data...")

try:
    test_file = "preprocessed_data_concept_level/sub-02/preprocessed_eeg_test.npy"
    test_data = np.load(test_file, allow_pickle=True).item()
    test_eeg = test_data['preprocessed_eeg_data']
    
    print(f"✓ Loaded test EEG: {test_file}")
    print(f"  Shape: {test_eeg.shape}")
    print(f"  Expected: (200, n_reps, 17, 250)")
    
    n_test_concepts = test_eeg.shape[0]
    n_test_reps = test_eeg.shape[1]
    
    print(f"\n  Test concepts: {n_test_concepts}")
    print(f"  Repetitions per concept: {n_test_reps}")
    
    if n_test_concepts != 200:
        print(f"\n✗✗✗ WRONG NUMBER OF TEST CONCEPTS!")
        print(f"  Expected: 200")
        print(f"  Got: {n_test_concepts}")
    
except Exception as e:
    print(f"✗ Could not load test EEG: {e}")
    print("\nTrying alternative path...")
    
    try:
        # Try original preprocessing output
        test_file = "/content/drive/MyDrive/Colab_Notebooks/preprocessed_data/sub-02/preprocessed_eeg_test.npy"
        test_data = np.load(test_file, allow_pickle=True).item()
        test_eeg = test_data['preprocessed_eeg_data']
        
        print(f"✓ Loaded from original path: {test_file}")
        print(f"  Shape: {test_eeg.shape}")
        
        n_test_concepts = test_eeg.shape[0]
        n_test_reps = test_eeg.shape[1]
    except:
        print("✗ Could not find test data anywhere!")
        sys.exit(1)

# =============================================================================
# 2. Check CLIP Test Embeddings
# =============================================================================

print("\n" + "="*70)
print("2. Checking CLIP Test Embeddings...")
print("="*70)

try:
    clip_all = np.load("THINGS_clip_embeddings/clip_embeddings_concept_level.npy")
    print(f"✓ Loaded CLIP embeddings: {clip_all.shape}")
    
    # Test embeddings should be at indices 1654-1853
    clip_test = clip_all[1654:1854]
    print(f"  Test CLIP subset: {clip_test.shape}")
    print(f"  Expected: (200, 768)")
    
    # Check norms
    norms = np.linalg.norm(clip_test, axis=1)
    print(f"\n  Test CLIP norms:")
    print(f"    Mean: {norms.mean():.4f}")
    print(f"    Min: {norms.min():.4f}")
    print(f"    Max: {norms.max():.4f}")
    
    if norms.mean() < 0.9 or norms.mean() > 1.1:
        print(f"  ⚠️  WARNING: CLIP norms not normalized!")
    
except Exception as e:
    print(f"✗ Could not load CLIP embeddings: {e}")
    sys.exit(1)

# =============================================================================
# 3. Simulate Test Evaluation Logic
# =============================================================================

print("\n" + "="*70)
print("3. Simulating Test Evaluation...")
print("="*70)

print("\nTest evaluation logic:")
print("  1. Load test EEG: (200, n_reps, 17, 250)")
print("  2. Data loader returns image_ids = 1654-1853 (global indices)")
print("  3. Evaluate function:")
print("     - Searches in CLIP[1654:1854]")
print("     - Gets predictions 0-199 (local in test subset)")
print("     - Adds 1654 offset → predictions become 1654-1853")
print("     - Compares with image_ids (1654-1853)")

print("\nChecking if image_ids match CLIP indices...")

# Simulate data loader
from torch.utils.data import Dataset

class TestDataCheck(Dataset):
    def __init__(self):
        self.eeg_data = test_eeg
        self.clip_embeddings = clip_all
        self.global_image_offset = 1654  # For test
    
    def __len__(self):
        return self.eeg_data.shape[0]
    
    def __getitem__(self, idx):
        global_img_idx = idx + self.global_image_offset
        return {
            'image_id': global_img_idx,
            'local_idx': idx
        }

test_check = TestDataCheck()

print(f"\nData loader returns:")
for i in [0, 1, 2, 199]:
    if i < len(test_check):
        sample = test_check[i]
        print(f"  Test concept {i} → image_id = {sample['image_id']}")

print(f"\nExpected image_ids: 1654, 1655, 1656, ..., 1853")

# Check if indices are correct
expected_ids = list(range(1654, 1854))
actual_ids = [test_check[i]['image_id'] for i in range(len(test_check))]

if actual_ids == expected_ids:
    print("\n✓✓✓ Image IDs are CORRECT!")
else:
    print("\n✗✗✗ Image IDs are WRONG!")
    print(f"  First 5 expected: {expected_ids[:5]}")
    print(f"  First 5 actual: {actual_ids[:5]}")

# =============================================================================
# 4. Check Test Data Quality
# =============================================================================

print("\n" + "="*70)
print("4. Test Data Quality Check...")
print("="*70)

# Average test EEG across repetitions
test_eeg_avg = test_eeg.mean(axis=1)  # (200, 17, 250)

print(f"\nAveraged test EEG: {test_eeg_avg.shape}")

# Check variance
variance_per_concept = test_eeg_avg.var(axis=(1, 2))
print(f"\nVariance per test concept:")
print(f"  Mean: {variance_per_concept.mean():.6f}")
print(f"  Min: {variance_per_concept.min():.6f}")
print(f"  Max: {variance_per_concept.max():.6f}")

if variance_per_concept.min() < 0.0001:
    print("\n✗✗✗ Some test concepts have near-zero variance!")
    print("  This suggests corrupted or collapsed data!")

# Check if all test concepts are different
print(f"\nChecking if test concepts are distinct...")
correlations = []
for i in range(min(10, len(test_eeg_avg))):
    for j in range(i+1, min(10, len(test_eeg_avg))):
        corr = np.corrcoef(test_eeg_avg[i].flatten(), test_eeg_avg[j].flatten())[0, 1]
        correlations.append(abs(corr))

mean_corr = np.mean(correlations) if correlations else 0
print(f"  Mean absolute correlation (first 10): {mean_corr:.4f}")

if mean_corr > 0.8:
    print("  ✗✗✗ Test concepts are too similar!")
else:
    print("  ✓ Test concepts are distinct")

# =============================================================================
# 5. Compare Train vs Test Data Distribution
# =============================================================================

print("\n" + "="*70)
print("5. Train vs Test Distribution...")
print("="*70)

try:
    train_file = "preprocessed_data_concept_level/sub-02/preprocessed_eeg_training.npy"
    train_data = np.load(train_file, allow_pickle=True).item()
    train_eeg = train_data['preprocessed_eeg_data']
    
    print(f"✓ Loaded training EEG: {train_eeg.shape}")
    
    # Compare statistics
    train_avg = train_eeg.mean(axis=1)  # Average reps
    
    print(f"\nTraining stats:")
    print(f"  Mean: {train_avg.mean():.6f}")
    print(f"  Std: {train_avg.std():.6f}")
    
    print(f"\nTest stats:")
    print(f"  Mean: {test_eeg_avg.mean():.6f}")
    print(f"  Std: {test_eeg_avg.std():.6f}")
    
    # Check if distributions are similar
    mean_diff = abs(train_avg.mean() - test_eeg_avg.mean())
    std_diff = abs(train_avg.std() - test_eeg_avg.std())
    
    print(f"\nDifferences:")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Std diff: {std_diff:.6f}")
    
    if mean_diff > 0.1 or std_diff > 0.1:
        print("\n⚠️  WARNING: Train and test distributions are different!")
        print("  This could cause poor generalization")
    
except Exception as e:
    print(f"Could not load training data: {e}")

# =============================================================================
# 6. Summary
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

issues = []

if n_test_concepts != 200:
    issues.append(f"Wrong number of test concepts ({n_test_concepts}, expected 200)")

if variance_per_concept.min() < 0.0001:
    issues.append("Some test concepts have near-zero variance (corrupted data)")

if mean_corr > 0.8:
    issues.append("Test concepts are too similar (not distinct)")

if norms.mean() < 0.9 or norms.mean() > 1.1:
    issues.append("CLIP test embeddings not normalized")

if len(issues) == 0:
    print("\n✓ No obvious test data issues detected!")
    print("\nPossible causes of 12% val vs 1.5% test:")
    print("  1. Model overfitting to training distribution")
    print("  2. Test data genuinely harder (200 unseen concepts)")
    print("  3. Need more training (model hasn't converged)")
    print("  4. Hyperparameter tuning needed")
    
    print("\nRecommendations:")
    print("  - Train longer (try 100-150 epochs)")
    print("  - Try lower learning rate (1e-5 instead of 1e-4)")
    print("  - Add more regularization (higher dropout)")
    print("  - Try different architecture variants")
else:
    print(f"\n✗ Found {len(issues)} issue(s) with test data:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\nThese issues might explain the 1.5% test accuracy!")

print("\n" + "="*70)
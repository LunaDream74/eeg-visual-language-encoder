"""
Diagnostic Script: Verify Everything is Correct

Run this to check if your data, model, and evaluation are aligned.
"""

import torch
import numpy as np
import pickle

print("="*70)
print("DIAGNOSTIC: CHECKING EVERYTHING")
print("="*70)

# =============================================================================
# 1. Check Preprocessed Data
# =============================================================================

print("\n1. CHECKING PREPROCESSED DATA")
print("-"*70)

with open('./preprocessed_data_250Hz/sub-02/preprocessed_eeg_training.npy', 'rb') as f:
    train_dict = pickle.load(f)
train_eeg = train_dict['preprocessed_eeg_data']

print(f"Training EEG shape: {train_eeg.shape}")
print(f"  Expected: (16540, 4, 17, 250)")
print(f"  Channels: {train_eeg.shape[2]}")

if train_eeg.shape[0] != 16540:
    print("  ❌ WRONG: Not 16,540 images!")
elif train_eeg.shape[2] != 17:
    print(f"  ❌ WRONG: Has {train_eeg.shape[2]} channels, not 17!")
else:
    print("  ✓ Correct!")

# =============================================================================
# 2. Check CLIP Embeddings
# =============================================================================

print("\n2. CHECKING CLIP EMBEDDINGS")
print("-"*70)

clip_emb = np.load('THINGS_clip_embeddings/clip_embeddings_image_level.npy')
print(f"CLIP embeddings shape: {clip_emb.shape}")
print(f"  Expected: (16740, 768)")

if clip_emb.shape[0] != 16740:
    print(f"  ❌ WRONG: Has {clip_emb.shape[0]} embeddings, not 16,740!")
else:
    print("  ✓ Correct!")

# Check normalization
norms = np.linalg.norm(clip_emb, axis=1)
print(f"\nCLIP embedding norms:")
print(f"  Mean: {norms.mean():.6f} (should be ~1.0)")
print(f"  Std: {norms.std():.6f} (should be ~0.0)")

if abs(norms.mean() - 1.0) > 0.1:
    print("  ⚠️  WARNING: CLIP embeddings not normalized!")
else:
    print("  ✓ Normalized correctly")

# =============================================================================
# 3. Test Data Loader
# =============================================================================

print("\n3. TESTING DATA LOADER")
print("-"*70)

from other_model.data_loader_image_level import create_dataloaders_image_level

train_loader, val_loader, test_loader = create_dataloaders_image_level(
    preprocessed_path='./preprocessed_data_250Hz',
    clip_embeddings_path='THINGS_clip_embeddings/clip_embeddings_image_level.npy',
    subject_id=2,
    batch_size=32
)

# Check a batch
for batch in train_loader:
    print(f"\nTrain batch:")
    print(f"  EEG shape: {batch['eeg'].shape}")
    print(f"  CLIP shape: {batch['clip_emb'].shape}")
    print(f"  Image IDs range: {batch['image_id'].min()}-{batch['image_id'].max()}")
    
    expected_channels = 17
    if batch['eeg'].shape[1] != expected_channels:
        print(f"  ❌ WRONG: EEG has {batch['eeg'].shape[1]} channels, expected {expected_channels}!")
    else:
        print(f"  ✓ EEG channels correct ({expected_channels})")
    
    if batch['image_id'].max() >= 16540:
        print(f"  ❌ WRONG: Train image IDs should be 0-16539!")
    else:
        print(f"  ✓ Image IDs correct (0-16539)")
    break

for batch in val_loader:
    print(f"\nValidation batch:")
    print(f"  Image IDs range: {batch['image_id'].min()}-{batch['image_id'].max()}")
    
    if batch['image_id'].max() >= 16540:
        print(f"  ❌ WRONG: Val image IDs should be 0-16539!")
    else:
        print(f"  ✓ Image IDs correct (0-16539)")
    break

for batch in test_loader:
    print(f"\nTest batch:")
    print(f"  Image IDs range: {batch['image_id'].min()}-{batch['image_id'].max()}")
    
    if batch['image_id'].min() < 16540:
        print(f"  ❌ WRONG: Test image IDs should be 16540-16739!")
    elif batch['image_id'].max() > 16739:
        print(f"  ❌ WRONG: Test image IDs should be 16540-16739!")
    else:
        print(f"  ✓ Image IDs correct (16540-16739)")
    break

# =============================================================================
# 4. Test Model
# =============================================================================

print("\n4. TESTING MODEL")
print("-"*70)

from old_mostly_infonce.simplified_architecture import create_model

model, loss_fn = create_model(
    architecture='simplified',
    n_channels=17,
    n_timepoints=250,
    latent_dim=768,
    dropout=0.3
)

# Forward pass
for batch in train_loader:
    eeg = batch['eeg']
    clip_target = batch['clip_emb']
    
    print(f"\nModel forward pass:")
    print(f"  Input shape: {eeg.shape}")
    
    try:
        output = model(eeg)
        print(f"  Output shape: {output.shape}")
        print(f"  ✓ Model forward pass works!")
        
        # Check loss
        loss, mse_loss, infonce_loss = loss_fn(output, clip_target)
        print(f"\nLoss values:")
        print(f"  Total: {loss.item():.4f}")
        print(f"  MSE: {mse_loss.item():.4f}")
        print(f"  InfoNCE: {infonce_loss.item():.4f}")
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    
    break

# =============================================================================
# 5. Test Evaluation Function
# =============================================================================

print("\n5. TESTING EVALUATION LOGIC")
print("-"*70)

# Simulate what evaluation does
val_batch = next(iter(val_loader))
image_ids = val_batch['image_id']

print(f"Validation image IDs sample: {image_ids[:5].tolist()}")
print(f"  Min: {image_ids.min()}, Max: {image_ids.max()}")

# Check retrieval range
print(f"\nRetrieval setup:")
print(f"  Searching against: clip_embeddings_all[:16540]")
print(f"  This gives embeddings for images: 0-16539 ✓")
print(f"  Validation image IDs range: {image_ids.min()}-{image_ids.max()}")

if image_ids.max() >= 16540:
    print(f"  ❌ PROBLEM: Some validation IDs >= 16540!")
else:
    print(f"  ✓ All validation IDs in correct range")

# =============================================================================
# 6. Manual Retrieval Test
# =============================================================================

print("\n6. MANUAL RETRIEVAL TEST")
print("-"*70)

# Get one validation example
val_batch = next(iter(val_loader))
eeg_sample = val_batch['eeg'][0:1]  # Take first sample
true_id = val_batch['image_id'][0].item()
true_clip = val_batch['clip_emb'][0]

print(f"Testing retrieval for image ID: {true_id}")

# Forward pass
with torch.no_grad():
    pred_emb = model(eeg_sample)
    pred_emb_norm = torch.nn.functional.normalize(pred_emb, dim=1)[0]

# Get all training CLIP embeddings
clip_train = clip_emb[:16540]  # All training images
clip_train_tensor = torch.tensor(clip_train, dtype=torch.float32)
clip_train_norm = torch.nn.functional.normalize(clip_train_tensor, dim=1)

# Compute similarities
similarities = torch.mm(pred_emb_norm.unsqueeze(0), clip_train_norm.t())[0]

# Get top matches
top5_indices = similarities.topk(5)[1].tolist()
top5_sims = similarities.topk(5)[0].tolist()

print(f"\nTop 5 matches:")
for i, (idx, sim) in enumerate(zip(top5_indices, top5_sims)):
    marker = "✓ CORRECT!" if idx == true_id else ""
    print(f"  {i+1}. Image {idx}: similarity {sim:.4f} {marker}")

if true_id in top5_indices:
    print(f"\n✓ Model can retrieve correct image in Top-5!")
    rank = top5_indices.index(true_id) + 1
    print(f"  Rank: {rank}/16540")
else:
    true_sim = similarities[true_id].item()
    print(f"\n✗ Correct image not in Top-5")
    print(f"  Correct image similarity: {true_sim:.4f}")
    print(f"  Rank: {(similarities > true_sim).sum().item() + 1}/16540")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
print("\nIf all checks passed (✓), the setup is correct.")
print("If you see ❌, fix those issues first!")
print("\nIf everything is ✓ but validation is still 0%,")
print("the problem is likely in the training script's evaluate() function.")

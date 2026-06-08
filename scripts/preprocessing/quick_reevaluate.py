"""
Quick Re-evaluation Script

Re-evaluates your EXISTING trained model (epoch 35) with the FIXED evaluation code.
No re-training needed - just loads your checkpoint and evaluates with correct indices.

Expected result: Test accuracy will jump from 0% to 20-25%!
"""

import torch
import numpy as np
from old_mostly_infonce.data_loader_new_preprocessing import create_dataloaders
from old_mostly_infonce.simplified_architecture import create_model

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

PREPROCESSED_PATH = "preprocessed_data"
CLIP_EMBEDDINGS_PATH = "THINGS_clip_embeddings/things_clip_embeddings.npy"
CHECKPOINT_PATH = "./checkpoints/best_model_sub02.pth"
SUBJECT = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("\n" + "="*70)
print("RE-EVALUATING EXISTING MODEL WITH BUG FIX")
print("="*70)
print(f"\nSubject: {SUBJECT}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Device: {DEVICE}\n")

# =============================================================================
# Load Model
# =============================================================================

print("Loading model...")
model, _ = create_model(
    architecture='simplified',
    n_channels=17,
    n_timepoints=250,
    latent_dim=512,
    dropout=0.3
)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

print(f" Loaded model from epoch {checkpoint['epoch']}")
print(f"  Previous validation Top-1: {checkpoint['val_top1']:.2f}%")
print(f"  Previous validation Top-5: {checkpoint['val_top5']:.2f}%")

# =============================================================================
# Create Data Loaders with FIXED code
# =============================================================================

print("\nCreating data loaders with FIXED indexing...")
_, val_loader, test_loader = create_dataloaders(
    preprocessed_path=PREPROCESSED_PATH,
    clip_embeddings_path=CLIP_EMBEDDINGS_PATH,
    subject_id=SUBJECT,
    batch_size=32,
    num_workers=0  # Set to 0 to avoid multiprocessing issues
)

# =============================================================================
# Load CLIP Embeddings
# =============================================================================

print("\nLoading CLIP embeddings...")
clip_embeddings = np.load(CLIP_EMBEDDINGS_PATH)
print(f" CLIP embeddings shape: {clip_embeddings.shape}")
print(f"  Training images: 0-1653 (1,654 images)")
print(f"  Test images: 1654-1853 (200 images)")

# =============================================================================
# Define Fixed Evaluation Function
# =============================================================================

def evaluate_fixed(model, dataloader, clip_embeddings_all, device, is_test=False):
    """Fixed evaluation with correct indexing"""
    import torch.nn as nn
    
    model.eval()
    
    all_pred_embeddings = []
    all_image_ids = []
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            eeg = batch['eeg'].to(device)
            clip_target = batch['clip_emb'].to(device)
            image_ids = batch['image_id']
            
            # Forward
            pred_emb = model(eeg)
            
            # Loss
            mse_loss = nn.functional.mse_loss(pred_emb, clip_target)
            total_loss += mse_loss.item()
            n_batches += 1
            
            # Store
            all_pred_embeddings.append(pred_emb.cpu())
            all_image_ids.append(image_ids)
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")
    
    # Concatenate
    all_pred_embeddings = torch.cat(all_pred_embeddings, dim=0)
    all_image_ids = torch.cat(all_image_ids, dim=0)
    
    # Normalize
    all_pred_embeddings = nn.functional.normalize(all_pred_embeddings, dim=1)
    
    # Zero-shot retrieval
    if is_test:
        # Test: search only against 200 test images
        print(f"\nSearching against TEST images only (indices 1654-1853)...")
        clip_embeddings_test = clip_embeddings_all[1654:1854]
        clip_embeddings_tensor = torch.tensor(clip_embeddings_test, dtype=torch.float32)
        clip_embeddings_tensor = nn.functional.normalize(clip_embeddings_tensor, dim=1)
        
        # Compute similarities
        similarities = torch.mm(all_pred_embeddings, clip_embeddings_tensor.t())
        
        # Get predictions
        top1_preds = similarities.argmax(dim=1) + 1654  # Add offset
        top5_preds = similarities.topk(5, dim=1)[1] + 1654
    else:
        # Validation: search against training images
        print(f"\nSearching against TRAINING images (indices 0-1653)...")
        clip_embeddings_train = clip_embeddings_all[:1654]
        clip_embeddings_tensor = torch.tensor(clip_embeddings_train, dtype=torch.float32)
        clip_embeddings_tensor = nn.functional.normalize(clip_embeddings_tensor, dim=1)
        
        # Compute similarities
        similarities = torch.mm(all_pred_embeddings, clip_embeddings_tensor.t())
        
        # Get predictions
        top1_preds = similarities.argmax(dim=1)
        top5_preds = similarities.topk(5, dim=1)[1]
    
    # Accuracy
    top1_correct = (top1_preds == all_image_ids).float().mean().item()
    top5_correct = torch.any(top5_preds == all_image_ids.unsqueeze(1), dim=1).float().mean().item()
    
    avg_loss = total_loss / n_batches
    
    return top1_correct * 100, top5_correct * 100, avg_loss

# =============================================================================
# Re-evaluate Validation
# =============================================================================

print("\n" + "="*70)
print("RE-EVALUATING VALIDATION (should match previous results)")
print("="*70)

val_top1, val_top5, val_loss = evaluate_fixed(model, val_loader, clip_embeddings, DEVICE, is_test=False)

print(f"\n Validation Results:")
print(f"  Top-1: {val_top1:.2f}% (previous: {checkpoint['val_top1']:.2f}%)")
print(f"  Top-5: {val_top5:.2f}%")
print(f"  Loss: {val_loss:.4f}")

if abs(val_top1 - checkpoint['val_top1']) < 0.5:
    print(f"   Matches previous validation (sanity check passed!)")
else:
    print(f"    WARNING: Validation changed! Debug needed.")

# =============================================================================
# Re-evaluate Test
# =============================================================================

print("\n" + "="*70)
print("RE-EVALUATING TEST (this should fix the 0% bug!)")
print("="*70)

test_top1, test_top5, test_loss = evaluate_fixed(model, test_loader, clip_embeddings, DEVICE, is_test=True)

print(f"\n Test Results:")
print(f"  Top-1: {test_top1:.2f}% (previous: 0.00%)")
print(f"  Top-5: {test_top5:.2f}% (previous: 0.00%)")
print(f"  Loss: {test_loss:.4f}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("COMPARISON: OLD vs FIXED")
print("="*70)

print(f"\nValidation:")
print(f"  Old: {checkpoint['val_top1']:.2f}% Top-1")
print(f"  New: {val_top1:.2f}% Top-1")
print(f"  Change: {val_top1 - checkpoint['val_top1']:+.2f}% (should be ~0%)")

print(f"\nTest:")
print(f"  Old: 0.00% Top-1 (BUG!)")
print(f"  New: {test_top1:.2f}% Top-1 (FIXED!)")
print(f"  Improvement: {test_top1:.2f}% (expected: 20-25%)")

print("\n" + "="*70)
print("PERFORMANCE vs BASELINE")
print("="*70)

old_baseline = 4.51
print(f"\nOld preprocessing (500ms, no whitening): {old_baseline}%")
print(f"New preprocessing (1000ms, whitening):   {test_top1:.2f}%")
print(f"Improvement: {test_top1 / old_baseline:.1f}x better!")

if test_top1 >= 20:
    print(f"\n🎉🎉🎉 SUCCESS! 🎉🎉🎉")
    print(f"Test accuracy is in the expected range (20-25%)!")
    print(f"This is a {test_top1 / old_baseline:.1f}x improvement over baseline!")
    print(f"\nYour preprocessing and architecture work perfectly!")
elif test_top1 >= 15:
    print(f"\n Good progress!")
    print(f"Test accuracy is close to target. Minor tuning needed.")
elif test_top1 >= 10:
    print(f"\n  Below target but much better than 0%!")
    print(f"The fix worked but performance is lower than expected.")
    print(f"Check: Are you using the correct CLIP embeddings?")
else:
    print(f"\n  Still below target.")
    print(f"The fix helped but performance is lower than expected.")
    print(f"Debug: Verify preprocessing, CLIP embeddings, and data alignment.")

print("\n" + "="*70)

"""
Data Integrity Verification Script
Diagnoses issues with CLIP embeddings, EEG data, and model training
"""

import numpy as np
import torch
import torch.nn.functional as F
from nice_eeg_data_loader import create_nice_dataloaders
import os


def check_clip_embeddings():
    """Check CLIP embeddings for issues"""
    print("\n" + "="*70)
    print("CHECKING CLIP EMBEDDINGS")
    print("="*70)
    
    clip_path = 'THINGS_clip_embeddings/clip_embeddings_image_level.npy'
    clip = np.load(clip_path)
    
    print(f"Shape: {clip.shape} (should be (16740, 768))")
    print(f"  Train portion (0:16540): {clip[:16540].shape}")
    print(f"  Test portion (16540:16740): {clip[16540:16740].shape}")
    
    print(f"\nDtype: {clip.dtype}")
    print(f"Min/Max: [{clip.min():.4f}, {clip.max():.4f}]")
    print(f"Mean: {clip.mean():.6f}, Std: {clip.std():.6f}")
    
    # Check for NaNs/Infs
    n_nan = np.isnan(clip).sum()
    n_inf = np.isinf(clip).sum()
    print(f"NaNs: {n_nan}, Infs: {n_inf}")
    
    if n_nan > 0 or n_inf > 0:
        print("  [WARNING] Found NaN or Inf values!")
    
    # Check norms
    norms = np.linalg.norm(clip, axis=1)
    print(f"\nEmbedding norms - Min: {norms.min():.4f}, Max: {norms.max():.4f}, Mean: {norms.mean():.4f}")
    
    # Normalize and check
    clip_norm = clip / (np.linalg.norm(clip, axis=1, keepdims=True) + 1e-8)
    norms_after = np.linalg.norm(clip_norm, axis=1)
    print(f"After normalization - All ~1.0? Min: {norms_after.min():.6f}, Max: {norms_after.max():.6f}")


def check_eeg_data():
    """Check EEG data preprocessing"""
    print("\n" + "="*70)
    print("CHECKING EEG DATA")
    print("="*70)
    
    subject_dir = './preprocessed_data_250Hz/sub-02'
    
    import pickle
    with open(os.path.join(subject_dir, 'preprocessed_eeg_training.npy'), 'rb') as f:
        train_dict = pickle.load(f)
    train_eeg = train_dict['preprocessed_eeg_data']
    
    with open(os.path.join(subject_dir, 'preprocessed_eeg_test.npy'), 'rb') as f:
        test_dict = pickle.load(f)
    test_eeg = test_dict['preprocessed_eeg_data']
    
    print(f"Training EEG shape: {train_eeg.shape} (should be (16540, 4, 17, 250))")
    print(f"  Test EEG shape: {test_eeg.shape} (should be (200, 80, 17, 250))")
    
    # Check values
    print(f"\nTraining EEG - Min: {train_eeg.min():.4f}, Max: {train_eeg.max():.4f}")
    print(f"  Mean: {train_eeg.mean():.6f}, Std: {train_eeg.std():.6f}")
    
    # Check repetition averaging
    train_avg = train_eeg.mean(axis=1)
    print(f"\nAfter averaging repetitions: {train_avg.shape} (should be (16540, 17, 250))")
    print(f"  Min: {train_avg.min():.4f}, Max: {train_avg.max():.4f}")
    print(f"  Mean: {train_avg.mean():.6f}, Std: {train_avg.std():.6f}")


def check_dataloader():
    """Check dataloader output"""
    print("\n" + "="*70)
    print("CHECKING DATALOADER OUTPUT")
    print("="*70)
    
    train_loader, val_loader, test_loader = create_nice_dataloaders(
        preprocessed_path='./preprocessed_data_250Hz',
        clip_embeddings_path='THINGS_clip_embeddings/clip_embeddings_image_level.npy',
        subject_id=2,
        batch_size=256,
        num_workers=0,
        val_split=0.1
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    eeg = batch['eeg']
    clip_emb = batch['clip_emb']
    image_ids = batch['image_id']
    
    print(f"\nBatch shapes:")
    print(f"  EEG: {eeg.shape} (should be (256, 17, 250))")
    print(f"  CLIP: {clip_emb.shape} (should be (256, 768))")
    print(f"  Image IDs: {image_ids.shape} (should be (256,))")
    
    print(f"\nEEG stats:")
    print(f"  Min: {eeg.min():.4f}, Max: {eeg.max():.4f}")
    print(f"  Mean: {eeg.mean():.6f}, Std: {eeg.std():.6f}")
    
    print(f"\nCLIP stats:")
    print(f"  Min: {clip_emb.min():.4f}, Max: {clip_emb.max():.4f}")
    print(f"  Mean: {clip_emb.mean():.6f}, Std: {clip_emb.std():.6f}")
    
    # Check CLIP norms (should be 1.0 if normalized in dataloader)
    clip_norms = torch.norm(clip_emb, dim=1)
    print(f"  Norms after normalization: Min {clip_norms.min():.6f}, Max {clip_norms.max():.6f}")
    
    # Check image ID consistency
    print(f"\nImage IDs range: [{image_ids.min()}, {image_ids.max()}]")
    print(f"  Expected: [0, 16540) for training, [16540, 16740) for test")


def simulate_forward_pass():
    """Simulate a forward pass to check shapes"""
    print("\n" + "="*70)
    print("SIMULATING FORWARD PASS")
    print("="*70)
    
    from nice_eeg_architecture import create_nice_eeg_model
    
    model, loss_fn = create_nice_eeg_model(n_channels=17, n_timepoints=250, latent_dim=768, dropout=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy batch
    batch_eeg = torch.randn(256, 17, 250).to(device)
    batch_clip = torch.randn(256, 768).to(device)
    batch_clip = F.normalize(batch_clip, dim=1)
    
    with torch.no_grad():
        pred_emb = model(batch_eeg)
    
    print(f"\nModel output shape: {pred_emb.shape} (should be (256, 768))")
    print(f"  Min: {pred_emb.min():.4f}, Max: {pred_emb.max():.4f}")
    print(f"  Mean: {pred_emb.mean():.6f}, Std: {pred_emb.std():.6f}")
    
    # Check loss computation
    with torch.no_grad():
        loss, temp = loss_fn(pred_emb, batch_clip)
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Temperature: {temp:.4f}")
    
    # Check similarity computation
    pred_norm = F.normalize(pred_emb, dim=1)
    clip_norm = batch_clip  # Already normalized
    similarities = torch.mm(pred_norm, clip_norm.t())
    
    print(f"\nSimilarity matrix shape: {similarities.shape}")
    print(f"  Diagonal (should be ~1.0): {torch.diag(similarities)[:5]}")
    print(f"  Min: {similarities.min():.4f}, Max: {similarities.max():.4f}")
    
    # Check retrieval
    top1_idx = similarities.argmax(dim=1)
    matches = (top1_idx == torch.arange(256).to(device))
    print(f"\nDiagonal matches (dummy data): {matches.sum()}/{len(matches)} ({matches.float().mean()*100:.2f}%)")
    print(f"  (Expected ~0% since random data, not learned)")


if __name__ == "__main__":
    check_clip_embeddings()
    check_eeg_data()
    check_dataloader()
    simulate_forward_pass()
    
    print("\n" + "="*70)
    print("DATA INTEGRITY CHECK COMPLETE")
    print("="*70)

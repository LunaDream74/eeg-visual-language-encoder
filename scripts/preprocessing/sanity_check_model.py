"""
Sanity Check: Verify Model is Learning from EEG

Tests if model performance drops when we:
1. Shuffle EEG-CLIP pairs (should drop to ~0%)
2. Use random noise instead of EEG (should drop to ~0%)
3. Permute EEG channels (should drop significantly)

If model maintains high accuracy after these perturbations,
it's NOT learning from EEG properly!
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os

from nice_eeg_architecture import create_nice_eeg_model
from nice_eeg_data_loader import create_nice_dataloaders


def sanity_check_1_shuffled_labels(model, test_loader, clip_embeddings_all, device):
    """
    Test 1: Shuffle EEG-CLIP pairs
    Expected: Accuracy should drop to ~0.5% (random chance)
    """
    print("\n" + "="*70)
    print("SANITY CHECK 1: Shuffled EEG-CLIP Pairs")
    print("="*70)
    print("Breaking the EEG → CLIP correspondence...")
    
    model.eval()
    all_pred = []
    all_shuffled_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch['eeg'].to(device)
            image_ids = batch['image_id']
            
            # Shuffle image IDs (breaks EEG-CLIP correspondence!)
            shuffled_ids = image_ids[torch.randperm(len(image_ids))]
            
            pred_emb = model(eeg)
            pred_emb = F.normalize(pred_emb, dim=1)
            
            all_pred.append(pred_emb.cpu())
            all_shuffled_ids.append(shuffled_ids)
    
    all_pred = torch.cat(all_pred, dim=0)
    all_shuffled_ids = torch.cat(all_shuffled_ids, dim=0)
    
    # Evaluate with shuffled labels
    clip_test = clip_embeddings_all[16540:16740]
    clip_test = torch.tensor(clip_test, dtype=torch.float32)
    clip_test = F.normalize(clip_test, dim=1)
    similarities = torch.mm(all_pred, clip_test.t())
    top1_preds = similarities.argmax(dim=1) + 16540
    
    accuracy = (top1_preds == all_shuffled_ids).float().mean().item() * 100
    
    print(f"\nAccuracy with SHUFFLED labels: {accuracy:.2f}%")
    print(f"Expected (random chance): ~0.5%")
    
    if accuracy > 2.0:
        print("⚠️  WARNING: Model accuracy too high with shuffled labels!")
        print("   Model might NOT be learning from EEG properly!")
        return False
    else:
        print("✓ PASS: Model accuracy dropped as expected")
        return True


def sanity_check_2_random_noise(model, test_loader, clip_embeddings_all, device):
    """
    Test 2: Replace EEG with random noise
    Expected: Accuracy should drop to ~0.5%
    """
    print("\n" + "="*70)
    print("SANITY CHECK 2: Random Noise Instead of EEG")
    print("="*70)
    print("Replacing EEG with Gaussian noise...")
    
    model.eval()
    all_pred = []
    all_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Replace EEG with random noise
            noise = torch.randn_like(batch['eeg']).to(device)
            image_ids = batch['image_id']
            
            pred_emb = model(noise)
            pred_emb = F.normalize(pred_emb, dim=1)
            
            all_pred.append(pred_emb.cpu())
            all_ids.append(image_ids)
    
    all_pred = torch.cat(all_pred, dim=0)
    all_ids = torch.cat(all_ids, dim=0)
    
    # Evaluate
    clip_test = clip_embeddings_all[16540:16740]
    clip_test = torch.tensor(clip_test, dtype=torch.float32)
    clip_test = F.normalize(clip_test, dim=1)
    similarities = torch.mm(all_pred, clip_test.t())
    top1_preds = similarities.argmax(dim=1) + 16540
    
    accuracy = (top1_preds == all_ids).float().mean().item() * 100
    
    print(f"\nAccuracy with RANDOM NOISE: {accuracy:.2f}%")
    print(f"Expected (random chance): ~0.5%")
    
    if accuracy > 2.0:
        print("⚠️  WARNING: Model accuracy too high with random noise!")
        print("   Model is NOT reading EEG signals!")
        return False
    else:
        print("✓ PASS: Model accuracy dropped as expected")
        return True


def sanity_check_3_permuted_channels(model, test_loader, clip_embeddings_all, device):
    """
    Test 3: Randomly permute EEG channels
    Expected: Accuracy should drop significantly (to 1-5%)
    """
    print("\n" + "="*70)
    print("SANITY CHECK 3: Permuted EEG Channels")
    print("="*70)
    print("Randomly shuffling spatial channel order...")
    
    model.eval()
    all_pred = []
    all_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch['eeg'].to(device)
            image_ids = batch['image_id']
            
            # Permute channels randomly
            perm = torch.randperm(eeg.shape[1])
            eeg_permuted = eeg[:, perm, :]
            
            pred_emb = model(eeg_permuted)
            pred_emb = F.normalize(pred_emb, dim=1)
            
            all_pred.append(pred_emb.cpu())
            all_ids.append(image_ids)
    
    all_pred = torch.cat(all_pred, dim=0)
    all_ids = torch.cat(all_ids, dim=0)
    
    # Evaluate
    clip_test = clip_embeddings_all[16540:16740]
    clip_test = torch.tensor(clip_test, dtype=torch.float32)
    clip_test = F.normalize(clip_test, dim=1)
    similarities = torch.mm(all_pred, clip_test.t())
    top1_preds = similarities.argmax(dim=1) + 16540
    
    accuracy = (top1_preds == all_ids).float().mean().item() * 100
    
    print(f"\nAccuracy with PERMUTED CHANNELS: {accuracy:.2f}%")
    print(f"Expected: 1-5% (much lower than normal)")
    
    if accuracy > 8.0:
        print("⚠️  WARNING: Model accuracy too high with permuted channels!")
        print("   Model might not be using spatial information properly!")
        return False
    else:
        print("✓ PASS: Model accuracy dropped as expected")
        return True


def evaluate_normal(model, test_loader, clip_embeddings_all, device):
    """Normal evaluation for comparison"""
    model.eval()
    all_pred = []
    all_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch['eeg'].to(device)
            image_ids = batch['image_id']
            
            pred_emb = model(eeg)
            pred_emb = F.normalize(pred_emb, dim=1)
            
            all_pred.append(pred_emb.cpu())
            all_ids.append(image_ids)
    
    all_pred = torch.cat(all_pred, dim=0)
    all_ids = torch.cat(all_ids, dim=0)
    
    clip_test = clip_embeddings_all[16540:16740]
    clip_test = torch.tensor(clip_test, dtype=torch.float32)
    clip_test = F.normalize(clip_test, dim=1)
    similarities = torch.mm(all_pred, clip_test.t())
    top1_preds = similarities.argmax(dim=1) + 16540
    
    accuracy = (top1_preds == all_ids).float().mean().item() * 100
    return accuracy


def main(args):
    print("\n" + "="*70)
    print("MODEL SANITY CHECKS")
    print("="*70)
    print("Verifying that model is actually learning from EEG signals...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load CLIP embeddings
    clip_embeddings_all = np.load(args.clip_embeddings_path)
    clip_dim = clip_embeddings_all.shape[1]
    
    # Load model
    model, _ = create_nice_eeg_model(latent_dim=clip_dim)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"\nLoaded checkpoint from: {args.checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint test Top-1: {checkpoint.get('test_top1', 'N/A'):.2f}%")
    
    # Create test loader
    _, _, test_loader = create_nice_dataloaders(
        preprocessed_path=args.preprocessed_path,
        clip_embeddings_path=args.clip_embeddings_path,
        subject_id=args.subject,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Baseline: Normal evaluation
    print("\n" + "="*70)
    print("BASELINE: Normal Evaluation")
    print("="*70)
    normal_acc = evaluate_normal(model, test_loader, clip_embeddings_all, device)
    print(f"Normal test accuracy: {normal_acc:.2f}%")
    
    # Run sanity checks
    results = []
    
    pass1 = sanity_check_1_shuffled_labels(model, test_loader, clip_embeddings_all, device)
    results.append(('Shuffled Labels', pass1))
    
    pass2 = sanity_check_2_random_noise(model, test_loader, clip_embeddings_all, device)
    results.append(('Random Noise', pass2))
    
    pass3 = sanity_check_3_permuted_channels(model, test_loader, clip_embeddings_all, device)
    results.append(('Permuted Channels', pass3))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Normal accuracy: {normal_acc:.2f}%\n")
    
    all_passed = all(p for _, p in results)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*70)
    
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("Model is learning from EEG signals correctly.")
        print("Safe to proceed with multi-subject training!")
    else:
        print("⚠️  SOME CHECKS FAILED!")
        print("Model might not be learning from EEG properly.")
        print("Investigate before scaling to multi-subject!")
    
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/best_nice_model_sub02.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--preprocessed_path', type=str, default='./preprocessed_data_250Hz')
    parser.add_argument('--clip_embeddings_path', type=str,
                       default='THINGS_clip_embeddings/clip_embeddings_image_level.npy')
    parser.add_argument('--subject', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=200)
    
    args = parser.parse_args()
    main(args)

"""
Diagnostic script to debug test evaluation

This will help identify why test accuracy is exactly at chance level

Usage:
    python diagnose_test_eval.py
    python diagnose_test_eval.py --checkpoint path/to/best_model.pth
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from old_mostly_infonce.data_loader_new_preprocessing import create_dataloaders
from simplified_architecture_v2 import create_model

def diagnose_test_evaluation(checkpoint_path=None):
    """Comprehensive diagnosis of test evaluation
    
    Args:
        checkpoint_path: Path to model checkpoint (should be the 56% validation model)
    """
    
    print("\n" + "="*70)
    print("DIAGNOSTIC: Test Evaluation Debug")
    print("="*70)
    
    # Load data
    preprocessed_path = './preprocessed_data_concept_level'
    clip_path = 'THINGS_clip_embeddings/clip_embeddings_concept_level.npy'
    subject = 2
    
    train_loader, val_loader, test_loader = create_dataloaders(
        preprocessed_path=preprocessed_path,
        clip_embeddings_path=clip_path,
        subject_id=subject,
        batch_size=32,
        num_workers=0
    )
    
    # Load CLIP embeddings
    clip_embeddings_all = np.load(clip_path)
    print(f"\n1. CLIP Embeddings Shape: {clip_embeddings_all.shape}")
    
    # Check CLIP embedding normalization
    clip_norms = np.linalg.norm(clip_embeddings_all, axis=1)
    print(f"\n2. CLIP Embedding Norms:")
    print(f"   Mean: {clip_norms.mean():.4f}")
    print(f"   Std: {clip_norms.std():.4f}")
    print(f"   Min: {clip_norms.min():.4f}")
    print(f"   Max: {clip_norms.max():.4f}")
    
    if clip_norms.mean() < 0.9 or clip_norms.mean() > 1.1:
        print(f"   ⚠️  WARNING: CLIP embeddings are NOT normalized!")
        print(f"   This will cause MSE-only training to fail!")
    else:
        print(f"   ✓ CLIP embeddings are normalized")
    
    # Check test data
    print(f"\n3. Test Data Check:")
    test_batch = next(iter(test_loader))
    print(f"   Batch EEG shape: {test_batch['eeg'].shape}")
    print(f"   Batch CLIP shape: {test_batch['clip_emb'].shape}")
    print(f"   Image IDs: {test_batch['image_id'][:5].tolist()}")
    print(f"   Expected range: 1654-1853")
    
    if test_batch['image_id'].min() < 1654 or test_batch['image_id'].max() > 1853:
        print(f"   ⚠️  WARNING: Test image IDs out of range!")
    else:
        print(f"   ✓ Test image IDs in correct range")
    
    # Check if test CLIP embeddings match
    print(f"\n4. Test CLIP Embedding Alignment:")
    test_clip_from_batch = test_batch['clip_emb'][0].numpy()
    test_img_id = test_batch['image_id'][0].item()
    test_clip_from_array = clip_embeddings_all[test_img_id]
    
    diff = np.abs(test_clip_from_batch - test_clip_from_array).mean()
    print(f"   Difference: {diff:.6f}")
    
    if diff > 0.001:
        print(f"   ⚠️  WARNING: Test CLIP embeddings don't match!")
        print(f"   This means data loading is broken!")
    else:
        print(f"   ✓ Test CLIP embeddings match correctly")
    
    # Load a trained model and check predictions
    print(f"\n5. Loading best model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use the original create_model from simplified_architecture.py
    # This will load with the original HybridLoss
    model, loss_fn = create_model(
        architecture='simplified',
        n_channels=17,
        n_timepoints=250,
        latent_dim=768,
        dropout=0.5
    )
    model = model.to(device)
    
    # Try multiple checkpoint paths
    if checkpoint_path:
        checkpoint_paths = [checkpoint_path]
    else:
        checkpoint_paths = [
            './checkpoints/best_model_sub02_v2_hybridloss.pth',  # Improved training
            # './checkpoints/best_model_sub02.pth',      # Original training
            # 'best_model_sub02.pth',                     # Current directory
        ]
    
    checkpoint_loaded = False
    for path in checkpoint_paths:
        try:
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✓ Loaded model from: {path}")
            print(f"   Epoch: {checkpoint['epoch']}")
            print(f"   Validation Top-1: {checkpoint.get('val_top1', 'N/A')}")
            
            # Verify this is the 56% validation model
            val_acc = checkpoint.get('val_top1', 0)
            if val_acc > 50:
                print(f"   ✓ This is the high-validation model (56%+) - CORRECT!")
            elif val_acc > 10:
                print(f"   ⚠️  This is a medium-validation model (~{val_acc:.1f}%)")
            else:
                print(f"   ⚠️  This is a low-validation model (~{val_acc:.1f}%) - might not be the right one")
            
            checkpoint_loaded = True
            break
        except Exception as e:
            if checkpoint_path:  # If user specified, show error
                print(f"   ✗ Failed to load {path}: {e}")
            continue
    
    if not checkpoint_loaded:
        print(f"   ⚠️  Could not load any model checkpoint!")
        print(f"   Please provide the path to your best model (56% validation)")
        print(f"   Or run: python diagnose_test_eval.py --checkpoint <path>")
        return
    
    # Get predictions for first test batch
    model.eval()
    with torch.no_grad():
        eeg = test_batch['eeg'].to(device)
        pred_emb = model(eeg)
        
        print(f"\n6. Model Predictions Analysis:")
        print(f"   Pred embeddings shape: {pred_emb.shape}")
        print(f"   Pred mean: {pred_emb.mean().item():.4f}")
        print(f"   Pred std: {pred_emb.std().item():.4f}")
        print(f"   Pred min: {pred_emb.min().item():.4f}")
        print(f"   Pred max: {pred_emb.max().item():.4f}")
        
        # Check if predictions are collapsed
        if pred_emb.std().item() < 0.01:
            print(f"   ⚠️  WARNING: Predictions collapsed to same value!")
        elif pred_emb.std().item() > 10.0:
            print(f"   ⚠️  WARNING: Predictions have extreme variance!")
        else:
            print(f"   ✓ Prediction variance looks reasonable")
        
        # Normalize predictions
        pred_norm = F.normalize(pred_emb, dim=1)
        
        # Get test CLIP embeddings
        clip_test = clip_embeddings_all[1654:1854]
        clip_test_tensor = torch.tensor(clip_test, dtype=torch.float32, device=device)
        clip_test_norm = F.normalize(clip_test_tensor, dim=1)
        
        # Compute similarities
        similarities = torch.mm(pred_norm, clip_test_norm.t())
        
        print(f"\n7. Similarity Matrix Analysis:")
        print(f"   Similarity shape: {similarities.shape}")
        print(f"   Similarity mean: {similarities.mean().item():.4f}")
        print(f"   Similarity std: {similarities.std().item():.4f}")
        print(f"   Similarity min: {similarities.min().item():.4f}")
        print(f"   Similarity max: {similarities.max().item():.4f}")
        
        # Get predictions
        top1_preds = similarities.argmax(dim=1) + 1654
        true_ids = test_batch['image_id'].to(device)
        
        print(f"\n8. Sample Predictions:")
        for i in range(min(10, len(true_ids))):
            true_id = true_ids[i].item()
            pred_id = top1_preds[i].item()
            max_sim = similarities[i].max().item()
            correct_sim = similarities[i, true_id - 1654].item()
            
            match = "✓" if pred_id == true_id else "✗"
            print(f"   {match} True: {true_id}, Pred: {pred_id}, "
                  f"MaxSim: {max_sim:.4f}, CorrectSim: {correct_sim:.4f}")
        
        # Check if all predictions are the same
        unique_preds = torch.unique(top1_preds)
        print(f"\n9. Prediction Diversity:")
        print(f"   Unique predictions: {len(unique_preds)} out of {len(top1_preds)}")
        
        if len(unique_preds) < 5:
            print(f"   ⚠️  WARNING: Model is predicting the same concept repeatedly!")
            print(f"   Most common predictions: {torch.bincount(top1_preds - 1654).topk(5)}")
        else:
            print(f"   ✓ Predictions are diverse")
        
        # Check for memorization vs learning
        print(f"\n10. Memorization Check:")
        print(f"   Testing if model learned CLIP space or just memorized training concepts...")
        
        # Compare similarities within test vs cross similarities
        test_self_sims = []
        for i in range(min(50, similarities.shape[0])):
            # Similarity to correct test concept
            correct_sim = similarities[i, true_ids[i] - 1654].item()
            test_self_sims.append(correct_sim)
        
        avg_correct_sim = np.mean(test_self_sims)
        avg_all_sim = similarities.mean().item()
        
        print(f"   Average similarity to correct concept: {avg_correct_sim:.4f}")
        print(f"   Average similarity to all concepts: {avg_all_sim:.4f}")
        print(f"   Difference: {avg_correct_sim - avg_all_sim:.4f}")
        
        if abs(avg_correct_sim - avg_all_sim) < 0.01:
            print(f"   ⚠️  CRITICAL: Model shows NO preference for correct concepts!")
            print(f"   This suggests the model memorized training concepts")
            print(f"   but learned NOTHING about the CLIP embedding space")
        elif avg_correct_sim > avg_all_sim + 0.05:
            print(f"   ✓ Model shows preference for correct concepts (good!)")
        else:
            print(f"   ⚠️  Model shows weak preference for correct concepts")
    
    print(f"\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose why balanced hybrid model gets 56% validation but 0.5% test"
    )
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default=None,
        help='Path to model checkpoint (the one with 56% validation)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CRITICAL: Diagnosing the 56% validation → 0.5% test mystery")
    print("="*70)
    
    diagnose_test_evaluation(checkpoint_path=args.checkpoint)
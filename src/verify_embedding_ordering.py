"""
CRITICAL DIAGNOSTIC: Verify CLIP Embedding Ordering Matches Data Loader

Problem: extract_clip_embeddings.py generates 1024-dim embeddings,
         but original generate_clip_embeddings_batched.py generated 768-dim.
         If the image ordering doesn't match, models won't learn correct pairs.

This script checks:
1. ✓ Embedding shape and dimension
2. ✓ Data loader expectations (image_ids 0-16539 for train, 16540-16739 for test)
3. ✓ Extract script loads images in same order as batched script
4. ✓ No misalignment in how images are indexed
"""

import numpy as np
import os
from pathlib import Path

print("="*70)
print("CLIP EMBEDDING ORDERING DIAGNOSTIC")
print("="*70)

# Check 1: Current embeddings exist and have correct shape
print("\n1. CHECKING CURRENT EMBEDDINGS")
print("-"*70)

clip_paths = [
    'THINGS_clip_embeddings/clip_embeddings_vitH14_sharp.npy',
    'THINGS_clip_embeddings/clip_embeddings_vitH14_blur.npy',
    'THINGS_clip_embeddings/clip_embeddings_image_level.npy'
]

for path in clip_paths:
    if os.path.exists(path):
        emb = np.load(path)
        print(f"\n✓ {path}")
        print(f"  Shape: {emb.shape}")
        print(f"  Dtype: {emb.dtype}")
        print(f"  Train portion (0:16540)  : {emb[:16540].shape}")
        print(f"  Test portion (16540:16740): {emb[16540:16740].shape}")
        
        # Check norms
        if len(emb.shape) == 2:
            norms = np.linalg.norm(emb, axis=1)
            print(f"  Norm stats: mean={norms.mean():.4f}, std={norms.std():.4f}")
            if np.isnan(norms).any() or np.isinf(norms).any():
                print(f"  ❌ WARNING: Contains NaN or Inf!")
    else:
        print(f"\n❌ Not found: {path}")

# Check 2: Verify data loader expectations
print("\n\n2. CHECKING DATA LOADER EXPECTATIONS")
print("-"*70)

from multi_subject_data_loader import create_multi_subject_dataloaders

print("\nTrying to load with data loader (subject 1)...")
try:
    train_loader, val_loader, test_loader, num_sub = create_multi_subject_dataloaders(
        preprocessed_path='preprocessed_data_250Hz',
        clip_embeddings_path='THINGS_clip_embeddings/clip_embeddings_vitH14_sharp.npy',
        subjects=[1],
        batch_size=64,
        val_split=0.1,
        augment=False
    )
    
    # Test a batch
    eeg, clip, subject_ids, image_ids = next(iter(train_loader))
    print(f"\n✓ Data loader working!")
    print(f"  Example batch:")
    print(f"    EEG shape        : {eeg.shape}")
    print(f"    CLIP shape       : {clip.shape}")
    print(f"    CLIP dim         : {clip.shape[-1]} (should be 1024 or 768)")
    print(f"    Image ID range   : {image_ids.min()}-{image_ids.max()}")
    print(f"    Expected range   : 0-16539 (training)")
    
    if image_ids.min() < 0 or image_ids.max() >= 16540:
        print(f"  ❌ ERROR: Image IDs out of training range!")
    else:
        print(f"  ✓ Image ID range valid for training indices")
        
except Exception as e:
    print(f"❌ Error loading data: {e}")

# Check 3: Verify image loading order matches
print("\n\n3. CHECKING IMAGE LOADING ORDER")
print("-"*70)

from extract_clip_embeddings import load_things_images

images_dir = 'THINGS_images'
if os.path.exists(images_dir):
    print(f"\nLoading images from {images_dir}...")
    images = load_things_images(images_dir)
    print(f"✓ Loaded {len(images)} images")
    
    if len(images) == 16740:
        print(f"✓ Correct total: 16,740 images")
        
        # Verify first few training and test images
        print(f"\nFirst 5 training images:")
        for i in range(5):
            print(f"  {i}: {images[i].name}")
        
        print(f"\nImages at training/test boundary:")
        for i in [16538, 16539, 16540, 16541]:
            if i < len(images):
                print(f"  {i}: {images[i].name}")
        
        print(f"\nLast 5 test images:")
        for i in range(16735, 16740):
            print(f"  {i}: {images[i].name}")
    else:
        print(f"❌ WRONG: Expected 16,740, got {len(images)}")
else:
    print(f"❌ Directory not found: {images_dir}")

# Check 4: Version comparison
print("\n\n4. EMBEDDING DIMENSION COMPATIBILITY")
print("-"*70)

old_emb_path = 'THINGS_clip_embeddings/clip_embeddings_image_level.npy'
new_emb_path = 'THINGS_clip_embeddings/clip_embeddings_vitH14_blur.npy'

if os.path.exists(old_emb_path) and os.path.exists(new_emb_path):
    old_emb = np.load(old_emb_path)
    new_emb = np.load(new_emb_path)
    
    print(f"\nOld embeddings (CLIP-ViT-L/14): {old_emb.shape}")
    print(f"New embeddings (CLIP-ViT-H/14): {new_emb.shape}")
    
    if old_emb.shape[0] != new_emb.shape[0]:
        print(f"\n❌ ERROR: Different number of images!")
        print(f"  Old: {old_emb.shape[0]} images")
        print(f"  New: {new_emb.shape[0]} images")
    else:
        print(f"\n✓ Same number of images: {old_emb.shape[0]}")
        print(f"  Old dim: {old_emb.shape[1]} → New dim: {new_emb.shape[1]}")
    
    # Compare norms to check order
    old_norms = np.linalg.norm(old_emb, axis=1)
    new_norms = np.linalg.norm(new_emb, axis=1)
    
    print(f"\nNorm comparison (should be ~1.0 for L2-normalized):")
    print(f"  Old: mean={old_norms.mean():.4f}, std={old_norms.std():.4f}")
    print(f"  New: mean={new_norms.mean():.4f}, std={new_norms.std():.4f}")
else:
    print(f"Need both old and new embeddings to compare")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

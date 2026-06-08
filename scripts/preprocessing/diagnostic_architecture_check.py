"""
ARCHITECTURE SANITY CHECK

Tests if the model architecture is fundamentally broken
"""

import torch
import numpy as np
from old_mostly_infonce.simplified_architecture import create_model

print("="*70)
print("ARCHITECTURE SANITY CHECK")
print("="*70)

# =============================================================================
# 1. Create model
# =============================================================================

print("\n1. CREATING MODEL")
print("-"*70)

model, loss_fn = create_model(
    architecture='simplified',
    n_channels=17,
    n_timepoints=250,
    latent_dim=512,
    dropout=0.3
)

print(f"✓ Model created")

# =============================================================================
# 2. Test forward pass
# =============================================================================

print("\n2. TESTING FORWARD PASS")
print("-"*70)

# Create dummy input
batch_size = 4
dummy_eeg = torch.randn(batch_size, 17, 250)
dummy_clip = torch.randn(batch_size, 512)

print(f"Input shape: {dummy_eeg.shape}")

# Forward pass
output = model(dummy_eeg)

print(f"Output shape: {output.shape}")
print(f"Expected: ({batch_size}, 512)")

if output.shape != (batch_size, 512):
    print(f"❌ WRONG OUTPUT SHAPE!")
else:
    print(f"✓ Output shape correct")

# Check output statistics
print(f"\nOutput statistics:")
print(f"  Mean: {output.mean().item():.4f}")
print(f"  Std: {output.std().item():.4f}")
print(f"  Min: {output.min().item():.4f}")
print(f"  Max: {output.max().item():.4f}")

# Check for NaN or Inf
if torch.isnan(output).any():
    print(f"  ❌ OUTPUT CONTAINS NaN!")
if torch.isinf(output).any():
    print(f"  ❌ OUTPUT CONTAINS Inf!")

# =============================================================================
# 3. Test loss function
# =============================================================================

print("\n3. TESTING LOSS FUNCTION")
print("-"*70)

total_loss, mse_loss, infonce_loss = loss_fn(output, dummy_clip)

print(f"Total loss: {total_loss.item():.4f}")
print(f"MSE loss: {mse_loss.item():.4f}")
print(f"InfoNCE loss: {infonce_loss.item():.4f}")

if torch.isnan(total_loss):
    print(f"  ❌ LOSS IS NaN!")

# =============================================================================
# 4. Test backward pass
# =============================================================================

print("\n4. TESTING BACKWARD PASS")
print("-"*70)

total_loss.backward()

# Check gradients
has_grad = False
zero_grad = False
nan_grad = False

for name, param in model.named_parameters():
    if param.grad is not None:
        has_grad = True
        if torch.isnan(param.grad).any():
            nan_grad = True
            print(f"  ❌ NaN gradient in {name}")
        if (param.grad == 0).all():
            zero_grad = True
            print(f"  ⚠️  All-zero gradient in {name}")

if has_grad:
    print(f"✓ Gradients computed")
else:
    print(f"❌ NO GRADIENTS!")

if nan_grad:
    print(f"❌ NaN gradients detected - training will fail")

# =============================================================================
# 5. Test with REAL data shape
# =============================================================================

print("\n5. TESTING WITH REALISTIC DATA")
print("-"*70)

# Create data that mimics preprocessed EEG (whitened, mean~0, std~1)
realistic_eeg = torch.randn(batch_size, 17, 250)  # Already mean=0, std=1
realistic_clip = torch.randn(batch_size, 512)
realistic_clip = torch.nn.functional.normalize(realistic_clip, dim=1)  # Normalize like CLIP

print(f"Realistic EEG:")
print(f"  Mean: {realistic_eeg.mean():.4f} (should be ~0)")
print(f"  Std: {realistic_eeg.std():.4f} (should be ~1)")

print(f"\nRealistic CLIP:")
print(f"  Norm: {torch.norm(realistic_clip, dim=1).mean():.4f} (should be ~1)")

# Forward
output_realistic = model(realistic_eeg)

print(f"\nOutput:")
print(f"  Mean: {output_realistic.mean().item():.4f}")
print(f"  Std: {output_realistic.std().item():.4f}")

# Normalize and compute similarity (like in evaluation)
output_norm = torch.nn.functional.normalize(output_realistic, dim=1)
similarities = torch.mm(output_norm, realistic_clip.t())

print(f"\nSimilarity matrix:")
print(f"  Shape: {similarities.shape}")
print(f"  Diagonal mean: {torch.diag(similarities).mean().item():.4f}")
print(f"  Off-diagonal mean: {similarities[~torch.eye(batch_size, dtype=bool)].mean().item():.4f}")

# Check if diagonal is higher (model should prefer correct pairs)
diagonal_mean = torch.diag(similarities).mean().item()
off_diagonal_mean = similarities[~torch.eye(batch_size, dtype=bool)].mean().item()

if diagonal_mean > off_diagonal_mean:
    print(f"  ✓ Diagonal > off-diagonal (correct pairs preferred)")
else:
    print(f"  ⚠️  Off-diagonal >= diagonal (random predictions)")

# =============================================================================
# 6. Load actual checkpoint and check
# =============================================================================

print("\n6. CHECKING TRAINED MODEL")
print("-"*70)

try:
    checkpoint = torch.load('./checkpoints/best_model_sub02.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Test on random data
    test_output = model(realistic_eeg)
    
    print(f"\nTrained model output:")
    print(f"  Mean: {test_output.mean().item():.4f}")
    print(f"  Std: {test_output.std().item():.4f}")
    
    # Check if all outputs are the same (collapsed model)
    output_var = test_output.var(dim=0).mean().item()
    print(f"  Variance across batch: {output_var:.6f}")
    
    if output_var < 0.001:
        print(f"  ❌ MODEL COLLAPSED! All outputs are nearly identical")
        print(f"     The model learned to output the same thing for all inputs")
    
except Exception as e:
    print(f"Could not load checkpoint: {e}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nIf you see:")
print("  ❌ Wrong shapes → Architecture bug")
print("  ❌ NaN or Inf → Numerical instability")
print("  ❌ No gradients → Broken backward pass")
print("  ❌ MODEL COLLAPSED → Training failed completely")

print("\n" + "="*70)

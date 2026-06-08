import numpy as np
import torch
import torch.nn.functional as F

# Load the CLIP embeddings used in training
clip_emb = np.load('THINGS_clip_embeddings/clip_embeddings_image_level.npy')
print(f"Total CLIP shape: {clip_emb.shape}")

# Check the split
clip_train = clip_emb[:16540]
clip_test = clip_emb[16540:16740]

print(f"Train CLIP shape: {clip_train.shape}")
print(f"Test CLIP shape: {clip_test.shape}")

# Now let's simulate what happens during validation evaluation
# Create random "predictions" from model
np.random.seed(42)
batch_size_val = 256
num_val_batches = 7

# Simulate predictions
all_pred = []
all_ids = []
for i in range(num_val_batches):
    batch_pred = np.random.randn(min(batch_size_val, 1654 - i*batch_size_val), 768).astype(np.float32)
    batch_pred = batch_pred / (np.linalg.norm(batch_pred, axis=1, keepdims=True) + 1e-8)
    
    # Simulate image IDs from validation set (random but in range [0-16540])
    num_samples = batch_pred.shape[0]
    batch_ids = np.random.randint(0, 16540, num_samples)
    
    all_pred.append(torch.tensor(batch_pred))
    all_ids.append(torch.tensor(batch_ids))

all_pred = torch.cat(all_pred, dim=0)
all_ids = torch.cat(all_ids, dim=0)

print(f"\nValidation predictions shape: {all_pred.shape}")
print(f"Validation IDs range: [{all_ids.min()}, {all_ids.max()}]")
print(f"Validation IDs unique: {len(set(all_ids.numpy()))}")

# Evaluate
clip_train_t = torch.tensor(clip_train, dtype=torch.float32)
clip_train_t = F.normalize(clip_train_t, dim=1)
similarities = torch.mm(all_pred, clip_train_t.t())
top1_preds = similarities.argmax(dim=1)

print(f"\nTop-1 predictions range: [{top1_preds.min()}, {top1_preds.max()}]")
matches = (top1_preds == all_ids)
print(f"Matches: {matches.sum()} / {len(matches)} = {matches.float().mean().item() * 100:.2f}%")

# The issue: are the IDs actually covering the right range?
print(f"\n=== DIAGNOSING ===")
print(f"all_ids actual values: {all_ids[:20].numpy()}")
print(f"top1_preds actual values: {top1_preds[:20].numpy()}")

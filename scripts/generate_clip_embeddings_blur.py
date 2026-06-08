"""
Generate UBP Fovea-Blurred CLIP Embeddings (768-dim, ViT-L/14)

Identical to generate_clip_embeddings_batched.py EXCEPT:
- Each image is fovea-blurred before CLIP encoding (UBP paper, blur radius 11)
- Output is clip_embeddings_blur_768.npy  (16740, 768)

Use for TRAINING only. Evaluate/test with the original sharp embeddings
(clip_embeddings_image_level.npy) — this is the correct UBP setup.

Blur prior intuition: the brain can't encode high-frequency image details
during 100ms RSVP. Blurred CLIP targets better match what EEG captures,
reducing the System GAP (UBP paper, Table 3: +8.9% over vanilla baseline).
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
from tqdm import tqdm
import os
import pickle

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAINING_IMAGES_DIR = "THINGS_images/training_images"
TEST_IMAGES_DIR     = "THINGS_images/test_images"
OUTPUT_PATH         = "clip_embeddings_blur_768.npy"        # ← blur output
CHECKPOINT_PATH     = "clip_embeddings_blur_checkpoint.pkl"  # ← separate checkpoint

BATCH_SIZE = 32

# UBP blur parameters (from paper Section 4.1 and Fig. 6)
BLUR_RADIUS  = 11    # peak performance in UBP sensitivity analysis
FOVEA_LAMBDA = 3.0   # decay rate: how quickly blur increases from centre

print("="*70)
print("GENERATING UBP FOVEA-BLURRED CLIP EMBEDDINGS (768-dim, ViT-L/14)")
print("="*70)
print(f"\nBatch size  : {BATCH_SIZE}")
print(f"Blur radius : {BLUR_RADIUS}  (UBP optimal)")
print(f"Output      : {OUTPUT_PATH}")
print("Note: Use these for TRAINING only. Evaluate with sharp embeddings.")

# =============================================================================
# Load CLIP Model
# =============================================================================

print("\n1. Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if device == "cpu":
    print("⚠️  WARNING: Using CPU - this will be slow!")
    print("   Estimated time: ~2 hours with batching")
    print("   Consider using GPU runtime for 30 min completion")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()

print("✓ Model loaded")

# =============================================================================
# Foveal Blur (UBP)
# =============================================================================

def make_fovea_alpha(h, w):
    """
    Blending mask alpha(i,j) = exp(-lambda * d(i,j) / L)
    Centre = 1.0 (fully sharp), periphery → 0.0 (fully blurred).
    """
    cy, cx = h / 2.0, w / 2.0
    ys = np.arange(h, dtype=np.float32) - cy
    xs = np.arange(w, dtype=np.float32) - cx
    dist = np.sqrt(ys[:, None]**2 + xs[None, :]**2)
    L = np.sqrt(cy**2 + cx**2)
    return np.exp(-FOVEA_LAMBDA * dist / L).astype(np.float32)


def apply_fovea_blur(img_pil):
    """
    Apply UBP foveal blur:
      blended = alpha * sharp + (1 - alpha) * gaussian_blurred
    Centre stays sharp; periphery gets blurred.
    Matches Eq. 4-5 from the UBP paper.
    """
    img  = np.array(img_pil, dtype=np.float32)
    blur = np.array(
        img_pil.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS)),
        dtype=np.float32
    )
    h, w = img.shape[:2]
    alpha   = make_fovea_alpha(h, w)[:, :, None]   # (H, W, 1) broadcast
    blended = alpha * img + (1 - alpha) * blur
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

# =============================================================================
# Helper Functions
# =============================================================================

def load_checkpoint():
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"\n✓ Loaded checkpoint: {checkpoint['processed']} images processed")
        return checkpoint
    return {'embeddings': [], 'processed': 0}

def save_checkpoint(embeddings, processed):
    """Save checkpoint"""
    checkpoint = {'embeddings': embeddings, 'processed': processed}
    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"  ✓ Checkpoint saved: {processed} images")

def process_batch(image_paths, batch_size=32):
    """Process a batch of images"""
    embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load images and apply foveal blur before CLIP encoding
        images = []
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img = apply_fovea_blur(img)   # ← UBP blur applied here
                images.append(img)
            except Exception as e:
                print(f"\n⚠️  Error loading {img_path}: {e}")
                images.append(Image.new('RGB', (224, 224)))
        
        # Process batch using the proper method (like in original script)
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use vision_model + visual_projection (same as original script)
            vision_outputs = model.vision_model(**inputs)
            image_features = vision_outputs[1]  # pooled output
            image_features = model.visual_projection(image_features)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        embeddings.extend(image_features.cpu().numpy())
    
    return embeddings

# =============================================================================
# Process Training Images
# =============================================================================

print("\n2. Processing training images...")

training_dir = Path(TRAINING_IMAGES_DIR)
concept_folders = sorted([d for d in training_dir.iterdir() if d.is_dir()])

print(f"Found {len(concept_folders)} concept folders")

# Collect all image paths first
print("Collecting image paths...")
all_image_paths = []
for concept_folder in tqdm(concept_folders, desc="Collecting paths"):
    image_files = sorted(list(concept_folder.glob("*.jpg")))
    all_image_paths.extend(image_files)

print(f"Total images: {len(all_image_paths)}")
print(f"Expected: 16,540")

# Load checkpoint if exists
checkpoint = load_checkpoint()
training_embeddings = checkpoint['embeddings']
processed = checkpoint['processed']

if processed > 0:
    print(f"Resuming from image {processed}/{len(all_image_paths)}")
    all_image_paths = all_image_paths[processed:]

# Process in batches with checkpointing
print(f"\nProcessing images in batches of {BATCH_SIZE}...")
CHECKPOINT_INTERVAL = 500  # Save checkpoint every 500 images

for i in tqdm(range(0, len(all_image_paths), BATCH_SIZE), desc="Processing batches"):
    batch_paths = all_image_paths[i:i+BATCH_SIZE]
    
    # Process batch
    batch_embeddings = process_batch(batch_paths, batch_size=BATCH_SIZE)
    training_embeddings.extend(batch_embeddings)
    
    processed += len(batch_paths)
    
    # Save checkpoint periodically
    if processed % CHECKPOINT_INTERVAL < BATCH_SIZE:
        save_checkpoint(training_embeddings, processed)

training_embeddings = np.array(training_embeddings)
print(f"\n✓ Training embeddings: {training_embeddings.shape}")

# =============================================================================
# Process Test Images
# =============================================================================

print("\n3. Processing test images...")

test_dir = Path(TEST_IMAGES_DIR)
test_concept_folders = sorted([d for d in test_dir.iterdir() if d.is_dir()])

print(f"Found {len(test_concept_folders)} test concepts")

# Collect test image paths (ONE per concept, sorted for consistency)
test_image_paths = []
for concept_folder in test_concept_folders:
    image_files = sorted(concept_folder.glob("*.jpg"))
    if image_files:
        test_image_paths.append(image_files[0])

print(f"Processing {len(test_image_paths)} test images...")
test_embeddings = process_batch(test_image_paths, batch_size=BATCH_SIZE)
test_embeddings = np.array(test_embeddings)

print(f"✓ Test embeddings: {test_embeddings.shape}")

# =============================================================================
# Combine and Save
# =============================================================================

print("\n4. Combining and saving...")

all_embeddings = np.concatenate([training_embeddings, test_embeddings], axis=0)

print(f"\nFinal shape: {all_embeddings.shape}")
print(f"  Expected: (16740, 768)")

assert all_embeddings.shape[0] == 16740, f"Expected 16740, got {all_embeddings.shape[0]}"
assert all_embeddings.shape[1] == 768,   f"Expected 768-dim, got {all_embeddings.shape[1]}"
print("✓ Shape correct!")

# Verify ordering against sharp embeddings
sharp_path = "clip_embeddings_image_level.npy"
if os.path.exists(sharp_path):
    print(f"\nVerifying ordering against sharp embeddings...")
    sharp = np.load(sharp_path)
    # Blurred and sharp embeddings of the SAME image should be nearest neighbours
    n_check = 20
    correct = 0
    for i in range(n_check):
        sims = (sharp @ all_embeddings[i])   # both already L2-normalised
        nn   = sims.argmax()
        if nn == i:
            correct += 1
    pct = correct / n_check * 100
    print(f"  Self-alignment: {correct}/{n_check} ({pct:.0f}%)")
    if pct >= 80:
        print("  ✓ Ordering matches sharp embeddings — safe to use for training")
    else:
        print("  ✗ ORDERING MISMATCH — do not use for training!")
        raise RuntimeError("Embedding ordering does not match sharp embeddings.")
else:
    print(f"\n⚠️  Sharp embeddings not found at {sharp_path} — skipping ordering check")

np.save(OUTPUT_PATH, all_embeddings)
print(f"\n✓ Saved to: {OUTPUT_PATH}")

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("✓ Cleaned up checkpoint file")

print("\n" + "="*70)
print("SUCCESS!")
print("="*70)
print(f"\n✅ Generated {all_embeddings.shape[0]} blurred image-level embeddings (768-dim)")
print(f"✅ Use for TRAINING: --clip_embeddings_path {OUTPUT_PATH}")
print(f"✅ Use for EVAL/TEST: --eval_clip_embeddings_path {sharp_path}")

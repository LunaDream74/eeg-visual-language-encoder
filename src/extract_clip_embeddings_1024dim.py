"""
Extract CLIP Embeddings for THINGS-EEG2

Generates TWO embedding files in one pass using ViT-H/14 (1024-dim):
  1. clip_embeddings_vitH14_sharp.npy  — original images (16740, 1024)
  2. clip_embeddings_vitH14_blur.npy   — fovea-blurred images (16740, 1024)

Blur prior from UBP paper:
  - Gaussian blur radius 11 (peak performance per their Fig. 6)
  - Foveal blending: sharp at centre, blurred toward periphery
  - Simulates the human visual system's limited peripheral resolution

CLIP model: ViT-H-14-laion2B-s32B-b79K (OpenCLIP)
  - 1024-dim output (vs 768-dim for ViT-L/14)
  - Used by both UBP and NeuroCLIP for best results

Usage:
    python extract_clip_embeddings.py \
        --images_dir /path/to/THINGS_images \
        --output_dir THINGS_clip_embeddings \
        --batch_size 64

Expected runtime: ~20-40 min on GPU, ~2-3 hours on CPU
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    raise ImportError("Run: pip install open-clip-torch")


# ── UBP blur parameters ──────────────────────────────────────────────────────

BLUR_RADIUS  = 11    # UBP Fig. 6: peak at radius 11
FOVEA_LAMBDA = 3.0   # controls how quickly blur increases from centre


def make_fovea_alpha(h, w):
    """
    Create blending mask alpha(i,j) = exp(-lambda * d(i,j) / L).
    Centre pixel gets alpha=1 (sharp), periphery approaches 0 (blurred).
    Returns float32 numpy array (H, W).
    """
    cy, cx = h / 2.0, w / 2.0
    ys = np.arange(h, dtype=np.float32) - cy
    xs = np.arange(w, dtype=np.float32) - cx
    dist = np.sqrt(ys[:, None]**2 + xs[None, :]**2)
    L = np.sqrt(cy**2 + cx**2)
    alpha = np.exp(-FOVEA_LAMBDA * dist / L).astype(np.float32)
    return alpha


def apply_fovea_blur(img_pil):
    """
    Apply UBP foveal blur to a PIL image.
    Returns a new PIL image blended between sharp centre and blurred periphery.
    """
    img  = np.array(img_pil, dtype=np.float32)
    blur = np.array(
        img_pil.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS)),
        dtype=np.float32
    )
    h, w = img.shape[:2]
    alpha = make_fovea_alpha(h, w)[:, :, None]   # broadcast over channels
    blended = alpha * img + (1 - alpha) * blur
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


# ── Main extraction ───────────────────────────────────────────────────────────

def load_things_images(images_dir):
    """
    Return sorted list of image paths matching THINGS-EEG2 ordering.
    
    CRITICAL: Must match the ordering from generate_clip_embeddings_batched.py
    to ensure correspondence with the original 768-dim embeddings.
    
    Handles THINGS_images folder structure:
      THINGS_images/
        ├── training_images/
        │   ├── 00001_aardvark/      (10 images per concept)
        │   ├── 00002_abacus/
        │   └── ... (up to 1654 concepts)
        └── test_images/
            ├── 00001_aircraft_carrier/ (1 image per concept)
            ├── ...                     (200 concepts)
    
    Ordering:
      1. All training images: sorted concepts × 10 images each (16,540 total)
      2. Test images: 1 per concept from sorted concepts (200 total)
    
    Total: 16,740 images = 16,540 training + 200 test
    """
    images_dir = Path(images_dir)
    image_paths = []
    
    # Check if training_images and test_images subdirectories exist
    training_dir = images_dir / "training_images"
    test_dir = images_dir / "test_images"
    
    if training_dir.exists() and test_dir.exists():
        # ─────────────────────────────────────────────────────────────
        # TRAINING IMAGES: all images, sorted by concept, then by image name
        # ─────────────────────────────────────────────────────────────
        training_concept_dirs = sorted([d for d in training_dir.iterdir() if d.is_dir()])
        
        for concept_dir in training_concept_dirs:
            # Get all image files in this concept folder, sorted
            image_files = sorted([
                f for f in concept_dir.iterdir() 
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            ])
            image_paths.extend(image_files)
        
        training_count = len(image_paths)
        
        # ─────────────────────────────────────────────────────────────
        # TEST IMAGES: ONE image per concept (FIRST SORTED), sorted by concept
        # ─────────────────────────────────────────────────────────────
        test_concept_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
        
        for concept_dir in test_concept_dirs:
            # Get image files SORTED to ensure consistent ordering
            image_files = sorted([
                f for f in concept_dir.iterdir() 
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            ])
            if image_files:
                # Take the first image (matches original script behavior)
                image_paths.append(image_files[0])
        
        test_count = len(image_paths) - training_count
        
        print(f"Found {training_count} training images + {test_count} test images = {len(image_paths)} total")
    else:
        # Fallback to old structure: expect concept folders directly in images_dir
        for concept_dir in sorted(images_dir.iterdir()):
            if not concept_dir.is_dir():
                continue
            for img_path in sorted(concept_dir.iterdir()):
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                    image_paths.append(img_path)
        
        print(f"Found {len(image_paths)} images in {images_dir}")
    
    return image_paths


def extract_embeddings(images_dir, output_dir, batch_size=64, device=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Load ViT-H/14 from OpenCLIP ──────────────────────────────────────────
    print("\nLoading ViT-H-14-laion2B-s32B-b79K from OpenCLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14',
        pretrained='laion2b_s32b_b79k'
    )
    model = model.to(device).eval()

    emb_dim = model.visual.output_dim
    print(f"Embedding dimension: {emb_dim}  (expected 1024)")

    # ── Collect image paths ───────────────────────────────────────────────────
    image_paths = load_things_images(images_dir)
    n_images    = len(image_paths)

    if n_images == 0:
        raise RuntimeError(f"No images found in {images_dir}. "
                           "Check --images_dir points to THINGS image folder.")

    # ── Allocate output arrays ────────────────────────────────────────────────
    sharp_embs = np.zeros((n_images, emb_dim), dtype=np.float32)
    blur_embs  = np.zeros((n_images, emb_dim), dtype=np.float32)

    # ── Extract in batches ────────────────────────────────────────────────────
    print(f"\nExtracting embeddings for {n_images} images (batch_size={batch_size})...")

    with torch.no_grad():
        for start in tqdm(range(0, n_images, batch_size)):
            end   = min(start + batch_size, n_images)
            batch = image_paths[start:end]

            sharp_tensors = []
            blur_tensors  = []

            for p in batch:
                try:
                    img = Image.open(p).convert('RGB')
                except Exception as e:
                    print(f"  Warning: could not open {p}: {e}  — using blank image")
                    img = Image.new('RGB', (224, 224), (128, 128, 128))

                sharp_tensors.append(preprocess(img))
                blur_tensors.append(preprocess(apply_fovea_blur(img)))

            sharp_batch = torch.stack(sharp_tensors).to(device)
            blur_batch  = torch.stack(blur_tensors).to(device)

            sharp_feat = model.encode_image(sharp_batch)
            blur_feat  = model.encode_image(blur_batch)

            # L2-normalise (unit norm)
            sharp_feat = F.normalize(sharp_feat, dim=1).cpu().numpy()
            blur_feat  = F.normalize(blur_feat,  dim=1).cpu().numpy()

            sharp_embs[start:end] = sharp_feat
            blur_embs[start:end]  = blur_feat

    # ── Save ──────────────────────────────────────────────────────────────────
    sharp_path = os.path.join(output_dir, 'clip_embeddings_vitH14_sharp.npy')
    blur_path  = os.path.join(output_dir, 'clip_embeddings_vitH14_blur.npy')

    np.save(sharp_path, sharp_embs)
    np.save(blur_path,  blur_embs)

    print(f"\nSaved:")
    print(f"  Sharp : {sharp_path}  {sharp_embs.shape}")
    print(f"  Blurred: {blur_path}  {blur_embs.shape}")
    print(f"\nNorm check (should be ~1.0):")
    print(f"  Sharp mean norm  : {np.linalg.norm(sharp_embs, axis=1).mean():.4f}")
    print(f"  Blurred mean norm: {np.linalg.norm(blur_embs,  axis=1).mean():.4f}")

    return sharp_path, blur_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir',  type=str, required=True,
                        help='Path to THINGS images directory')
    parser.add_argument('--output_dir',  type=str,
                        default='THINGS_clip_embeddings',
                        help='Where to save the .npy embedding files')
    parser.add_argument('--batch_size',  type=int, default=64,
                        help='Batch size for CLIP encoding (reduce if OOM)')
    parser.add_argument('--device',      type=str, default=None,
                        help='cuda / cpu (auto-detected if not set)')
    args = parser.parse_args()

    extract_embeddings(
        images_dir  = args.images_dir,
        output_dir  = args.output_dir,
        batch_size  = args.batch_size,
        device      = args.device
    )

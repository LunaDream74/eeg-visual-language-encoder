# ============================================================
# THINGS-EEG2 Caption Generation Script (Google Colab)
#
# Run this on Colab (free T4 GPU) to caption all 16,740
# THINGS training images + 200 test images with LLaVA-1.5-7b.
#
# HOW TO USE:
#   1. Upload your THINGS images folder to Google Drive:
#      MyDrive/THINGS/training_images/  (1654 concept folders)
#      MyDrive/THINGS/test_images/      (200 concept folders)
#   2. Open a new Colab notebook (Runtime -> T4 GPU)
#   3. Copy each cell (marked # %%) into a separate Colab cell
#   4. Run cells top to bottom
#   5. Download things_captions.json when done
#
# OUTPUT: things_captions.json
#   Keys "0".."16739" -> one-sentence caption string
#   Indices 0-16539 = training images (matches clip_embeddings_image_level.npy ordering)
#   Indices 16540-16739 = test images
#
# Estimated time on T4: ~3-4 hours for all 16,740 images
# Checkpoints saved every 500 images - safe to resume after disconnect
# ============================================================


# %% [Cell 1] Install dependencies
# ---------------------------------
# !pip install -q transformers accelerate pillow bitsandbytes


# %% [Cell 2] Mount Google Drive
# --------------------------------
from google.colab import drive
drive.mount('/content/drive')

TRAINING_DIR = "/content/drive/MyDrive/THINGS/training_images"
TEST_DIR = "/content/drive/MyDrive/THINGS/test_images"
CHECKPOINT_PATH = "/content/drive/MyDrive/THINGS/captions_checkpoint.json"
OUTPUT_PATH = "/content/drive/MyDrive/THINGS/things_captions.json"

# Verify directories exist
import os
assert os.path.isdir(TRAINING_DIR), f"Training dir not found: {TRAINING_DIR}"
assert os.path.isdir(TEST_DIR), f"Test dir not found: {TEST_DIR}"
print(f"Training dir: {TRAINING_DIR}")
print(f"Test dir:     {TEST_DIR}")


# %% [Cell 3] Collect image paths (CRITICAL - must match CLIP embedding order)
# ------------------------------------------------------------------------------
# This ordering MUST match generate_clip_embeddings_blur.py exactly:
#   - sorted() on concept folders
#   - sorted() on *.jpg within each folder
#   - training first (indices 0-16539), test after (indices 16540-16739)
#   - test: only the FIRST sorted jpg per concept folder
from pathlib import Path

def collect_image_paths(training_dir, test_dir):
    """
    Collect all image paths in the exact order that matches
    clip_embeddings_image_level.npy.
    """
    paths = []
    indices = []

    # Training images: 1654 concepts x 10 images = 16540 total
    training_path = Path(training_dir)
    concept_folders = sorted([d for d in training_path.iterdir() if d.is_dir()])
    idx = 0
    for folder in concept_folders:
        image_files = sorted(list(folder.glob("*.jpg")))
        for img_path in image_files:
            paths.append(img_path)
            indices.append(idx)
            idx += 1

    assert len(paths) == 16540, f"Expected 16540 training images, got {len(paths)}"
    print(f"Training images collected: {len(paths)}")
    print(f"  First: {paths[0]}")
    print(f"  Last:  {paths[-1]}")

    # Test images: 200 concepts x 1 image = 200 total
    test_path = Path(test_dir)
    test_folders = sorted([d for d in test_path.iterdir() if d.is_dir()])
    for folder in test_folders:
        image_files = sorted(list(folder.glob("*.jpg")))
        if image_files:
            paths.append(image_files[0])
            indices.append(idx)
            idx += 1

    assert len(paths) == 16740, f"Expected 16740 total images, got {len(paths)}"
    print(f"Test images collected:     {len(paths) - 16540}")
    print(f"Total images:              {len(paths)}")

    return paths, indices

all_image_paths, all_indices = collect_image_paths(TRAINING_DIR, TEST_DIR)


# %% [Cell 4] Load LLaVA-1.5-7b model
# --------------------------------------
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

print(f"Loading {MODEL_ID} ...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
print(f"Model loaded. Device map: {model.hf_device_map}")
print(f"GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# %% [Cell 5] Caption generation function
# -----------------------------------------
from PIL import Image

PROMPT = "USER: <image>\nDescribe the main object in this image in one brief sentence. Be concise.\nASSISTANT:"

def generate_caption(image_path, max_new_tokens=60):
    """Generate a single-sentence caption for one image."""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=PROMPT,
            images=image,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the newly generated tokens (after input length)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        caption = processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Keep only first sentence if model is verbose
        for sep in ['. ', '.\n', '! ', '? ']:
            if sep in caption:
                caption = caption.split(sep)[0] + '.'
                break

        return caption

    except Exception as e:
        print(f"  ERROR on {image_path}: {e}")
        return ""


# %% [Cell 6] Load checkpoint (resume after disconnect)
# -------------------------------------------------------
import json

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            data = json.load(f)
        print(f"Resuming from checkpoint: {len(data)} captions already done")
        return data
    return {}

def save_checkpoint(captions_dict):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(captions_dict, f)

captions = load_checkpoint()
print(f"Starting from index: {len(captions)}")


# %% [Cell 7] Main captioning loop
# ----------------------------------
# Checkpoints to Drive every 500 images. Safe to interrupt and resume.
from tqdm.auto import tqdm

CHECKPOINT_INTERVAL = 500

remaining_pairs = [
    (path, idx) for path, idx in zip(all_image_paths, all_indices)
    if str(idx) not in captions
]

print(f"Images remaining: {len(remaining_pairs)} / {len(all_image_paths)}")

for i, (img_path, img_idx) in enumerate(tqdm(remaining_pairs, desc="Captioning")):
    caption = generate_caption(img_path)
    captions[str(img_idx)] = caption

    # Save checkpoint periodically
    if (i + 1) % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(captions)
        print(f"\n  Checkpoint saved ({len(captions)} done). Sample: [{img_idx}] {caption}")

# Final save
save_checkpoint(captions)
print(f"\nDone! {len(captions)} captions saved to checkpoint.")


# %% [Cell 8] Validate and save final output
# --------------------------------------------
print("\n=== Validation ===")
print(f"Total captions: {len(captions)}")
print(f"Expected:       16740")
assert len(captions) == 16740, f"Missing captions! Only {len(captions)}/16740"

# Check no empty captions
empty = [k for k, v in captions.items() if not v.strip()]
if empty:
    print(f"WARNING: {len(empty)} empty captions at indices: {empty[:10]}")
else:
    print("No empty captions.")

# Spot-check first/last/mid
for check_idx in [0, 1, 10, 100, 1000, 8270, 16539, 16540, 16739]:
    print(f"  [{check_idx:5d}] {captions[str(check_idx)]}")

# Save final output
with open(OUTPUT_PATH, 'w') as f:
    json.dump(captions, f, indent=2)
print(f"\nSaved to: {OUTPUT_PATH}")
print("Download this file to your local machine.")


# %% [Cell 9] Optional: Visual spot-check
# -----------------------------------------
# Uncomment to visually inspect a sample of captioned images
#
# import matplotlib.pyplot as plt
# import random
#
# sample_indices = random.sample(range(16540), 6)
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# for ax, idx in zip(axes.flat, sample_indices):
#     img = Image.open(all_image_paths[idx])
#     ax.imshow(img)
#     ax.set_title(f"[{idx}] {captions[str(idx)]}", wrap=True, fontsize=9)
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

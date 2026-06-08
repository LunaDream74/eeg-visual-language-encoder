"""
THINGS-EEG2 Caption Generation — Local Script

Default: Qwen2-VL-2B-Instruct in 4-bit  →  ~1.5 GB VRAM  (P2000 / any GPU)

When you have access to the RTX A4000 (16 GB), switch to 7B for better captions:
  - Change MODEL_ID to "Qwen/Qwen2-VL-7B-Instruct"
  - Run with --fp16 (fp16 fits in 16 GB, no quantization loss)
  - Or keep --quantize for 7B-4bit (~5 GB, also fits A4000)

Usage (P2000 — default):
  python generate_captions_local.py

Usage (A4000 — higher quality):
  python generate_captions_local.py --model Qwen/Qwen2-VL-7B-Instruct --fp16

  # Still want 4-bit on A4000 to save headroom:
  python generate_captions_local.py --model Qwen/Qwen2-VL-7B-Instruct

Resume from partial checkpoint:
  python generate_captions_local.py --resume_from captions_checkpoint.json

Override image dirs:
  python generate_captions_local.py \\
    --training_dir C:/Users/Precision/EEGencoder/images_things/training_images \\
    --test_dir C:/Users/Precision/EEGencoder/images_things/test_images

Output:
  things_captions.json       — final output, keys "0".."16739"
  captions_checkpoint.json   — rolling checkpoint every 200 images (resume-safe)

VRAM guide:
  Qwen2-VL-2B  4-bit  ~1.5 GB  ← default, P2000 safe
  Qwen2-VL-2B  fp16   ~4.0 GB  ← also fits P2000 (try if 4-bit quality too low)
  Qwen2-VL-7B  4-bit  ~5.0 GB  ← fits P2000 but very tight; use on A4000 instead
  Qwen2-VL-7B  fp16  ~16.0 GB  ← A4000 fp16, best quality
"""

import os
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Defaults — change MODEL_ID here or pass --model on the CLI
# ---------------------------------------------------------------------------

# P2000 default: 2B in 4-bit (~1.5 GB VRAM)
DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

# A4000 upgrade: uncomment this line and comment the one above when you have
# access to the A4000. Run with --fp16 for best quality (16 GB), or keep
# 4-bit (--quantize default) to save VRAM headroom for larger batches.
# DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

DEFAULT_TRAINING_DIR = r"C:\Users\Precision\EEGencoder\images_things\training_images"
DEFAULT_TEST_DIR     = r"C:\Users\Precision\EEGencoder\images_things\test_images"
DEFAULT_OUTPUT       = "things_captions.json"
DEFAULT_CHECKPOINT   = "captions_checkpoint.json"
CHECKPOINT_INTERVAL  = 200


# ---------------------------------------------------------------------------
# Image path collection — must match clip_embeddings_image_level.npy order
# (Replicates generate_clip_embeddings_blur.py lines 162-173, 212-222)
# ---------------------------------------------------------------------------

def collect_image_paths(training_dir: str, test_dir: str):
    """
    Collect all 16,740 image paths in the exact order that matches
    clip_embeddings_image_level.npy.

    Returns:
        paths  : list of Path objects (16,740)
        indices: list of ints 0..16739
    """
    paths = []

    # Training: sorted concept folders, sorted jpgs within each
    train_path = Path(training_dir)
    concept_folders = sorted([d for d in train_path.iterdir() if d.is_dir()])
    for folder in concept_folders:
        for img in sorted(folder.glob("*.jpg")):
            paths.append(img)

    assert len(paths) == 16540, \
        f"Expected 16540 training images, found {len(paths)}. Check --training_dir."

    # Test: sorted concept folders, FIRST sorted jpg per concept only
    test_path = Path(test_dir)
    test_folders = sorted([d for d in test_path.iterdir() if d.is_dir()])
    for folder in test_folders:
        imgs = sorted(folder.glob("*.jpg"))
        if imgs:
            paths.append(imgs[0])

    assert len(paths) == 16740, \
        f"Expected 16740 total images, found {len(paths)}. Check --test_dir."

    return paths, list(range(len(paths)))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, fp16: bool):
    """
    Load model in 4-bit (default, P2000-safe) or fp16 (A4000 recommended for 7B).

    4-bit: uses bitsandbytes NF4 + double quantization — works on any GPU
    fp16:  pass --fp16 flag; ideal for Qwen2-VL-7B on A4000 (16 GB)
    """
    mode_str = "fp16" if fp16 else "4-bit NF4"
    print(f"Loading {model_id} ({mode_str})...")

    if fp16:
        # A4000 path — no quantization, best quality
        # For Qwen2-VL-7B-Instruct this needs ~16 GB VRAM (fits A4000, not P2000)
        # For Qwen2-VL-2B-Instruct this needs ~4 GB VRAM (fits P2000 too)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Default path — 4-bit quantization, P2000-safe
        # 2B-4bit: ~1.5 GB  |  7B-4bit: ~5 GB (tight on P2000, fine on A4000)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # saves ~0.4 GB extra
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    # Cap image resolution to 256x256 equivalent (256*256 = 65536 pixels).
    # Default Qwen2-VL uses native resolution → very slow on P2000 (~40s/image).
    # At 65536 max_pixels: ~2-3s/image on P2000, quality still good for THINGS objects.
    # On A4000 you can raise to 512*512 = 262144 for better quality at ~1s/image.
    # Image resolution tradeoff:
    #   P2000  : 128x128 (16384 px)  → ~3-4s/image, ~14-18h for 16k images
    #   P2000  : 256x256 (65536 px)  → ~10s/image,  ~46h
    #   A4000  : 448x448 (200704 px) → ~0.5-1s/image — use this when on A4000
    max_px = 128 * 128  # P2000 default
    # max_px = 448 * 448  # uncomment for A4000
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        min_pixels=64 * 64,
        max_pixels=max_px,
    )

    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU after model load: {used:.1f} / {total:.1f} GB")

    return model, processor


# ---------------------------------------------------------------------------
# Single-image captioning
# ---------------------------------------------------------------------------

PROMPT = "Describe the main object in this image in one brief sentence. Be concise."

def generate_caption(model, processor, image_path: Path, device) -> str:
    try:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
            )

        # Decode only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        caption = processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Trim to first sentence if verbose
        for sep in ['. ', '.\n', '! ', '? ']:
            if sep in caption:
                caption = caption.split(sep)[0].rstrip('.') + '.'
                break

        return caption

    except Exception as e:
        print(f"\n  ERROR on {image_path.name}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str) -> dict:
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        print(f"Resuming from checkpoint: {len(data)} captions already done")
        return data
    return {}


def save_checkpoint(captions: dict, checkpoint_path: str):
    with open(checkpoint_path, 'w') as f:
        json.dump(captions, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print("=" * 70)
    print("THINGS-EEG2 CAPTION GENERATION (Local)")
    print("=" * 70)

    mode_str = "fp16" if args.fp16 else "4-bit NF4 (quantized)"
    print(f"Model  : {args.model}")
    print(f"Mode   : {mode_str}")
    print(f"Output : {args.output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}  ({props.total_memory/1e9:.1f} GB)")

        # Warn if trying to run 7B fp16 on a card with <12 GB
        if "7B" in args.model and args.fp16 and props.total_memory / 1e9 < 12:
            print("WARNING: 7B fp16 needs ~16 GB. Consider --model Qwen/Qwen2-VL-2B-Instruct")
            print("         or drop --fp16 to use 4-bit instead.")

    # Collect image paths
    print("\nCollecting image paths...")
    all_paths, all_indices = collect_image_paths(args.training_dir, args.test_dir)
    print(f"Total images: {len(all_paths)}")
    print(f"  First: {all_paths[0]}")
    print(f"  Last : {all_paths[-1]}")

    # Load existing captions (resume support)
    captions = load_checkpoint(args.resume_from)

    # Filter to remaining images
    remaining = [
        (path, idx) for path, idx in zip(all_paths, all_indices)
        if str(idx) not in captions
    ]
    print(f"\nRemaining: {len(remaining)} / {len(all_paths)}")

    if not remaining:
        print("All images already captioned. Saving final output.")
    else:
        model, processor = load_model(args.model, args.fp16)

        print(f"\nCaptioning {len(remaining)} images...")
        print(f"Checkpoint every {CHECKPOINT_INTERVAL} images -> {args.checkpoint}\n")

        for i, (img_path, img_idx) in enumerate(tqdm(remaining, desc="Captioning")):
            caption = generate_caption(model, processor, img_path, device)
            captions[str(img_idx)] = caption

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(captions, args.checkpoint)
                tqdm.write(f"  [{i+1}/{len(remaining)}] checkpoint saved "
                           f"({len(captions)} total) | last: {caption[:60]}")

        # Final checkpoint before validation
        save_checkpoint(captions, args.checkpoint)
        print(f"\nCaptioning complete. {len(captions)} captions generated.")

    # Validate
    print("\n=== Validation ===")
    print(f"Total captions : {len(captions)}")
    assert len(captions) == 16740, \
        f"Expected 16740, got {len(captions)}. Some images may have failed."

    empty = [k for k, v in captions.items() if not v.strip()]
    if empty:
        print(f"WARNING: {len(empty)} empty captions at indices: {empty[:10]}")
        print("Re-run to retry failed images (empty ones are skipped on resume).")
    else:
        print("No empty captions.")

    # Spot-check
    print("\nSpot-check (first/last/mid):")
    for idx in [0, 10, 100, 1000, 8270, 16539, 16540, 16739]:
        print(f"  [{idx:5d}] {captions[str(idx)]}")

    # Save final output
    with open(args.output, 'w') as f:
        json.dump(captions, f, indent=2)
    print(f"\nFinal output saved to: {args.output}")
    print("Ready for Stage 2 training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate THINGS image captions locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # P2000 — 2B 4-bit (default, ~1.5 GB VRAM):
  python generate_captions_local.py

  # A4000 — 7B fp16 (best quality, ~16 GB VRAM):
  python generate_captions_local.py --model Qwen/Qwen2-VL-7B-Instruct --fp16

  # A4000 — 7B 4-bit (good quality, ~5 GB VRAM):
  python generate_captions_local.py --model Qwen/Qwen2-VL-7B-Instruct

  # Resume from a partial run:
  python generate_captions_local.py --resume_from captions_checkpoint.json
        """
    )

    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='HuggingFace model ID. '
                             'Default: Qwen/Qwen2-VL-2B-Instruct (P2000). '
                             'Use Qwen/Qwen2-VL-7B-Instruct on A4000.')
    parser.add_argument('--fp16', action='store_true',
                        help='Load in fp16 instead of 4-bit. '
                             'For 2B: needs ~4 GB (fits P2000). '
                             'For 7B: needs ~16 GB (A4000 only).')
    parser.add_argument('--training_dir', type=str, default=DEFAULT_TRAINING_DIR)
    parser.add_argument('--test_dir',     type=str, default=DEFAULT_TEST_DIR)
    parser.add_argument('--output',       type=str, default=DEFAULT_OUTPUT,
                        help='Final output JSON path')
    parser.add_argument('--checkpoint',   type=str, default=DEFAULT_CHECKPOINT,
                        help='Rolling checkpoint path (saves every 200 images)')
    parser.add_argument('--resume_from',  type=str, default=None,
                        help='Partial captions JSON to resume from '
                             '(Colab checkpoint or previous local run)')

    args = parser.parse_args()
    main(args)

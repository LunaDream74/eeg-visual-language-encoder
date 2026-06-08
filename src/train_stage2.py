"""
Stage 2 Training: CLIP -> LLM Projector

Trains a single Linear(768, llm_embed_dim) projector that maps CLIP image
embeddings into the frozen LLM's token embedding space.  The LLM stays frozen
throughout; only the projector parameters update.

Following Thought2Text (Mishra et al., 2025):
  - 5 epochs, lr=2e-5, batch=16
  - Input: (CLIP embedding, caption) pairs
  - Projected CLIP embedding prepended as token 0
  - Cross-entropy loss on caption tokens only

Default LLM: microsoft/Phi-3.5-mini-instruct  (3.8B, ~2.5GB in 4-bit)
Fits on 5GB GPU.  To use LLaMA-3-8B with more GPU:
  --llm_name meta-llama/Meta-Llama-3-8B-Instruct

Usage:
  python train_stage2.py \\
    --llm_name microsoft/Phi-3.5-mini-instruct \\
    --captions_path things_captions.json \\
    --batch_size 16 --epochs 5 --lr 2e-5 \\
    --output_dir ./checkpoints_stage2
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from stage2_projector import CLIPtoLLMProjector, save_projector
from stage2_data_loader import create_stage2_dataloaders


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def build_inputs_with_projected_clip(
    projector,
    llm,
    clip_emb,
    input_ids,
    attention_mask,
    labels,
):
    """
    Prepend the projected CLIP embedding as token 0.

    Returns:
        inputs_embeds  : (B, 1+L, D)
        attention_mask : (B, 1+L)
        labels         : (B, 1+L)   — -100 at position 0 (CLIP token)
    """
    B = clip_emb.shape[0]
    device = clip_emb.device

    # Project CLIP -> LLM embed dim: (B, D)
    projected = projector(clip_emb)

    # Get text token embeddings from frozen LLM: (B, L, D)
    token_embeds = llm.get_input_embeddings()(input_ids)

    # Match dtype to LLM embeddings (4-bit model computes in fp16)
    projected = projected.to(token_embeds.dtype)

    # Prepend projected embedding: (B, 1+L, D)
    projected_token = projected.unsqueeze(1)              # (B, 1, D)
    inputs_embeds = torch.cat([projected_token, token_embeds], dim=1)

    # Extend attention mask: prepend 1 for the CLIP token
    clip_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=device)
    attention_mask = torch.cat([clip_mask, attention_mask], dim=1)

    # Extend labels: prepend -100 (no loss on CLIP token position)
    clip_label = torch.full((B, 1), -100, dtype=labels.dtype, device=device)
    labels = torch.cat([clip_label, labels], dim=1)

    return inputs_embeds, attention_mask, labels


def train_epoch(projector, llm, train_loader, optimizer, device,
                epoch, grad_accum_steps,
                eeg_embeddings=None, clip_mix_ratio=0.3, eeg_noise_std=0.0):
    """
    eeg_embeddings : torch.Tensor (16540, 768) on CPU, or None for CLIP-only mode.
    clip_mix_ratio : fraction of each batch that keeps CLIP embeddings (rest uses EEG).
    eeg_noise_std  : std of Gaussian noise added to EEG embeddings (0 = disabled).
                     Applied after the EEG swap, before the forward pass.
                     Re-normalizes to unit sphere after adding noise.
    """
    projector.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        clip_emb       = batch['clip_emb'].to(device)
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        # Mixed EEG/CLIP: swap ~(1 - clip_mix_ratio) fraction of embeddings with EEG
        if eeg_embeddings is not None:
            image_idx = batch['image_idx']          # (B,) CPU long, range 0..16539
            B = clip_emb.shape[0]
            # True = use EEG  (~70%),  False = keep CLIP  (~30%)
            eeg_mask = torch.rand(B) > clip_mix_ratio
            if eeg_mask.any():
                eeg_idx = image_idx[eeg_mask]
                clip_emb = clip_emb.clone()
                eeg_selected = eeg_embeddings[eeg_idx].to(device)
                # Noise augmentation: add Gaussian noise then re-normalize
                if eeg_noise_std > 0.0:
                    noise = torch.randn_like(eeg_selected) * eeg_noise_std
                    eeg_selected = F.normalize(eeg_selected + noise, dim=1)
                clip_emb[eeg_mask] = eeg_selected

        inputs_embeds, attention_mask_ext, labels_ext = build_inputs_with_projected_clip(
            projector, llm, clip_emb, input_ids, attention_mask, labels
        )

        outputs = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_ext,
            labels=labels_ext,
        )
        loss = outputs.loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}]  "
                  f"Loss: {outputs.loss.item():.4f}")

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate_loss(projector, llm, val_loader, device):
    projector.eval()
    total_loss = 0.0

    for batch in val_loader:
        clip_emb       = batch['clip_emb'].to(device)
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        inputs_embeds, attention_mask_ext, labels_ext = build_inputs_with_projected_clip(
            projector, llm, clip_emb, input_ids, attention_mask, labels
        )

        outputs = llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_ext,
            labels=labels_ext,
        )
        total_loss += outputs.loss.item()

    return total_loss / len(val_loader)


# ---------------------------------------------------------------------------
# Training result logging (adapted from training_logger.py)
# ---------------------------------------------------------------------------

def log_result(log_file_txt, log_file_json, epoch, train_loss, val_loss,
               llm_name, batch_size, lr, notes):
    entry = (
        f"Epoch {epoch:3d} | "
        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
        f"LLM={llm_name}  bs={batch_size}  lr={lr:.2e}"
        + (f" | {notes}" if notes else "")
    )
    with open(log_file_txt, 'a') as f:
        f.write(entry + '\n')

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "llm_name": llm_name,
        "batch_size": batch_size,
        "lr": lr,
        "notes": notes or "",
    }
    records = []
    if os.path.exists(log_file_json):
        try:
            with open(log_file_json, 'r') as f:
                content = f.read().strip()
                if content:
                    records = json.loads(content)
        except (json.JSONDecodeError, Exception):
            records = []
    records.append(record)
    with open(log_file_json, 'w') as f:
        json.dump(records, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print("\n" + "=" * 70)
    print("STAGE 2: CLIP -> LLM PROJECTOR TRAINING")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  ({props.total_memory / 1e9:.1f} GB)")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer: {args.llm_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # ------------------------------------------------------------------
    # Load LLM (4-bit quantized, frozen)
    # ------------------------------------------------------------------
    print(f"\nLoading LLM (4-bit): {args.llm_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    # Detect LLM embed dim automatically — works for any HF causal LM
    llm_embed_dim = llm.get_input_embeddings().weight.shape[1]
    print(f"LLM embed dim: {llm_embed_dim}")

    if device.type == 'cuda':
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU after LLM load: {used:.2f}/{total:.1f} GB")

    # ------------------------------------------------------------------
    # Create projector (only trainable component)
    # ------------------------------------------------------------------
    projector = CLIPtoLLMProjector(clip_dim=768, llm_embed_dim=llm_embed_dim,
                                   projector_type=args.projector_type)
    projector = projector.to(device)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        projector.load_state_dict(ckpt['projector_state_dict'])
        print(f"Resumed projector weights from: {args.resume_from}")

    # ------------------------------------------------------------------
    # Load EEG embeddings for mixed training (optional)
    # ------------------------------------------------------------------
    eeg_embeddings = None
    if args.eeg_embeddings_path:
        print(f"\nLoading EEG embeddings: {args.eeg_embeddings_path}")
        eeg_embs_np = np.load(args.eeg_embeddings_path)        # (16540, 768)
        eeg_embeddings = torch.tensor(eeg_embs_np, dtype=torch.float32)
        print(f"  Shape: {eeg_embeddings.shape}  "
              f"clip_mix_ratio={args.clip_mix_ratio}  "
              f"(~{100*(1-args.clip_mix_ratio):.0f}% EEG per batch)  "
              f"noise_std={args.eeg_noise_std}")

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    train_loader, val_loader, _ = create_stage2_dataloaders(
        clip_embeddings_path=args.clip_embeddings_path,
        captions_path=args.captions_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        val_split=args.val_split,
        max_length=args.max_length,
        num_workers=args.num_workers,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
    )

    # ------------------------------------------------------------------
    # Optimizer (projector params only)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    print(f"\nOptimizer: AdamW  lr={args.lr:.2e}  weight_decay=0.01")
    print(f"Trainable params: {sum(p.numel() for p in projector.parameters()):,}")
    print(f"Frozen LLM params: {sum(p.numel() for p in llm.parameters()):,}")
    print(f"Gradient accumulation steps: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    # When resuming, seed best_val_loss from the last completed epoch's val_loss
    # so the best_projector.pth isn't overwritten with a worse checkpoint.
    best_val_loss = float('inf')
    if args.resume_from:
        best_val_loss = 1.4252647036227628  # epoch 4 val_loss from training_results.json
    patience_counter = 0

    log_txt  = os.path.join(args.output_dir, 'training_results.txt')
    log_json = os.path.join(args.output_dir, 'training_results.json')

    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)

    for epoch in range(args.start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  "
              f"(LR: {optimizer.param_groups[0]['lr']:.2e})")

        train_loss = train_epoch(
            projector, llm, train_loader, optimizer, device,
            epoch, args.grad_accum_steps,
            eeg_embeddings=eeg_embeddings, clip_mix_ratio=args.clip_mix_ratio,
            eeg_noise_std=args.eeg_noise_std,
        )
        val_loss = evaluate_loss(projector, llm, val_loader, device)

        print(f"Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}")

        log_result(
            log_txt, log_json,
            epoch, train_loss, val_loss,
            args.llm_name, args.batch_size, args.lr, args.notes,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_projector(
                projector,
                os.path.join(args.output_dir, 'best_projector.pth'),
                metadata={
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'llm_name': args.llm_name,
                    'llm_embed_dim': llm_embed_dim,
                    'clip_dim': 768,
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                    'notes': args.notes or '',
                },
            )
            print(f"  ** NEW BEST  val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Projector saved to: {args.output_dir}/best_projector.pth")
    print(f"\nNext step: python eval_stage2.py "
          f"--projector_path {args.output_dir}/best_projector.pth "
          f"--llm_name {args.llm_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Train CLIP->LLM projector")

    # Model
    parser.add_argument('--llm_name', type=str,
                        default='microsoft/Phi-3.5-mini-instruct',
                        help='HuggingFace LLM model ID. '
                             'P2000 (5GB): Phi-3.5-mini or Qwen2.5-1.5B. '
                             'Larger GPU: meta-llama/Meta-Llama-3-8B-Instruct')

    # Data
    parser.add_argument('--clip_embeddings_path', type=str,
                        default='THINGS_clip_embeddings/clip_embeddings_image_level.npy')
    parser.add_argument('--captions_path', type=str,
                        default='things_captions.json')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max token length (prompt + caption)')

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation. Effective batch = batch_size x this.')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--val_split', type=float, default=0.1)

    # Prompts
    parser.add_argument('--system_prompt', type=str,
                        default='You are a helpful vision assistant.')
    parser.add_argument('--user_prompt', type=str,
                        default='Describe this image in one sentence.')

    # Mixed EEG/CLIP training
    parser.add_argument('--eeg_embeddings_path', type=str, default=None,
                        help='Path to eeg_embeddings_train.npy (16540, 768). '
                             'If provided, mixes EEG and CLIP embeddings during training.')
    parser.add_argument('--clip_mix_ratio', type=float, default=0.3,
                        help='Fraction of each batch that uses CLIP embeddings '
                             '(remainder uses EEG). Default 0.3 = 70%% EEG / 30%% CLIP.')
    parser.add_argument('--eeg_noise_std', type=float, default=0.0,
                        help='Std of Gaussian noise added to EEG embeddings during training '
                             '(re-normalized after). 0=disabled. Try 0.05-0.1.')
    parser.add_argument('--projector_type', type=str, default='linear',
                        choices=['linear', 'mlp'],
                        help='Projector architecture. linear=single Linear (Thought2Text), '
                             'mlp=2-layer MLP with skip+LayerNorm.')

    # Resume
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to projector checkpoint to resume from '
                             '(e.g. checkpoints_stage2/best_projector.pth)')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='Epoch to start from (use with --resume_from)')

    # I/O
    parser.add_argument('--output_dir', type=str, default='./checkpoints_stage2')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--notes', type=str, default=None)

    args = parser.parse_args()
    main(args)

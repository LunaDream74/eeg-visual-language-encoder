"""
Masked Self-Supervised Pretraining for the Multi-Subject EEG Encoder
====================================================================

Why
---
The contrastive stage only ever sees ~16,540 labelled EEG-image pairs per
subject, and image labels are a weak signal for a noisy modality like EEG.
Recent state-of-the-art EEG decoders (LaBraM, MindAlign, etc.) close most of
the gap by *pretraining the encoder on unlabelled EEG first*, using a masked
reconstruction objective, and only then aligning to CLIP. This script does the
"first" part.

How
---
MAE-style masked reconstruction:

    full EEG (B,17,250)
      -> randomly zero out time patches (mask_ratio of the signal)
      -> encoder.encode_backbone(masked, sid) -> xEEG (B, nz)
      -> small decoder -> reconstructed EEG (B,17,250)
      -> MSE on the MASKED positions only

The decoder is a throwaway head. After pretraining we save ONLY the encoder's
state_dict, so the model you later ship for contrastive/retrieval still has the
same ~1.93M parameters — no inference-time cost. The contrastive trainer loads
these weights via its new --pretrained_path flag.

Run
---
    python masked_pretrain.py \
        --preprocessed_path ./preprocessed_data_250Hz \
        --subjects all \
        --epochs 40 --batch_size 256 --mask_ratio 0.5 \
        --output pretrained_encoder.pth

Then fine-tune with alignment:

    python train_multi_subject.py \
        --pretrained_path pretrained_encoder.pth \
        --subjects all --epochs 200 ...
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multi_subject_architecture import MultiSubjectNICEEEGEncoder
from pretrain_data_loader import create_pretrain_dataloader


# ----------------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------------
def random_time_mask(eeg, mask_ratio, patch_len):
    """
    Zero out random contiguous time patches, shared across channels.

    Masking whole time-patches (rather than scattered samples) forces the
    encoder to infer missing signal from temporal context instead of trivially
    interpolating neighbours — the same reason image MAE masks patches, not
    pixels. The mask is shared across the 17 channels at each time patch.

    Args:
        eeg        : (B, C, T) float tensor
        mask_ratio : fraction of time patches to hide (e.g. 0.5)
        patch_len  : length of each time patch (T should divide evenly; any
                     remainder stays visible)
    Returns:
        masked_eeg : (B, C, T) with masked patches set to 0
        mask       : (B, 1, T) bool, True where hidden (for the loss)
    """
    B, C, T = eeg.shape
    n_patches = T // patch_len
    n_mask = max(1, int(round(mask_ratio * n_patches)))

    mask = torch.zeros(B, 1, T, dtype=torch.bool, device=eeg.device)
    for b in range(B):
        perm = torch.randperm(n_patches, device=eeg.device)[:n_mask]
        for p in perm:
            start = int(p) * patch_len
            mask[b, 0, start:start + patch_len] = True

    masked_eeg = eeg.masked_fill(mask, 0.0)
    return masked_eeg, mask


# ----------------------------------------------------------------------------
# Pretraining model = shared encoder backbone + throwaway decoder
# ----------------------------------------------------------------------------
class MaskedEEGPretrainer(nn.Module):
    """
    Wraps a real MultiSubjectNICEEEGEncoder so the pretrained weights drop
    straight into the contrastive model. Only `self.encoder` is kept afterward;
    `self.decoder` is discarded.
    """

    def __init__(self, encoder, n_channels=17, n_timepoints=250,
                 decoder_hidden=512):
        super().__init__()
        self.encoder = encoder
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        out_dim = n_channels * n_timepoints

        self.decoder = nn.Sequential(
            nn.Linear(encoder.nz_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, out_dim),
        )

        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"\nMasked pretrainer:")
        print(f"  Encoder params (kept)     : {enc_params:,}")
        print(f"  Decoder params (discarded): {dec_params:,}")

    def forward(self, masked_eeg, subject_ids):
        x = self.encoder.encode_backbone(masked_eeg, subject_ids)  # (B, nz)
        recon = self.decoder(x)                                    # (B, C*T)
        recon = recon.view(-1, self.n_channels, self.n_timepoints)
        return recon


def masked_reconstruction_loss(recon, target, mask):
    """MSE computed only on masked time positions (MAE-style)."""
    full_mask = mask.expand_as(target)            # (B, C, T)
    diff2 = (recon - target) ** 2
    masked = diff2[full_mask]
    if masked.numel() == 0:
        return diff2.mean()
    return masked.mean()


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def main(args):
    print("\n" + "=" * 70)
    print("MASKED SELF-SUPERVISED EEG PRETRAINING")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.subjects == 'all':
        subjects = list(range(1, 11))
    else:
        subjects = [int(s) for s in args.subjects.split(',')]

    loader, num_subjects = create_pretrain_dataloader(
        preprocessed_path=args.preprocessed_path,
        subjects=subjects,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_reps=args.max_reps,
    )

    # Build a real encoder (same class/param names as the contrastive model)
    encoder = MultiSubjectNICEEEGEncoder(
        n_channels=17,
        n_timepoints=250,
        latent_dim=768,
        num_subjects=num_subjects,
        use_subject_embedding=True,
        subject_emb_dim=args.subject_emb_dim,
        nz_dim=args.nz_dim,
        dropout=0.5,
    )
    model = MaskedEEGPretrainer(
        encoder,
        n_channels=17,
        n_timepoints=250,
        decoder_hidden=args.decoder_hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    warmup_epochs = min(5, max(1, args.epochs // 10))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nOptimization:")
    print(f"  Optimizer   : AdamW (wd={args.weight_decay})")
    print(f"  LR          : {args.lr:.2e}  (warmup {warmup_epochs} ep, cosine)")
    print(f"  Mask ratio  : {args.mask_ratio}  (patch_len={args.patch_len})")
    print(f"  Epochs      : {args.epochs}")

    best_loss = float('inf')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch_idx, batch in enumerate(loader):
            eeg = batch['eeg'].to(device)               # (B, 17, 250)
            sid = batch['subject_id'].to(device)

            masked_eeg, mask = random_time_mask(
                eeg, args.mask_ratio, args.patch_len)

            recon = model(masked_eeg, sid)
            loss = masked_reconstruction_loss(recon, eeg, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += loss.item()
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                      f"recon MSE: {loss.item():.4f}")

        scheduler.step()
        epoch_loss = running / len(loader)
        print(f"Epoch {epoch}/{args.epochs}  recon MSE: {epoch_loss:.4f}  "
              f"(LR {optimizer.param_groups[0]['lr']:.2e})")

        # Save the ENCODER ONLY (decoder is throwaway). Saving the full encoder
        # state_dict means the contrastive trainer can load it with strict=False
        # and pick up every shared-backbone weight by name.
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    'epoch': epoch,
                    'recon_loss': epoch_loss,
                    'encoder_state_dict': model.encoder.state_dict(),
                    'subjects': subjects,
                    'nz_dim': args.nz_dim,
                    'subject_emb_dim': args.subject_emb_dim,
                },
                args.output,
            )
            print(f"  ✓ saved best encoder -> {args.output} "
                  f"(MSE {epoch_loss:.4f})")

    print("\nDone. Pretrained encoder at:", args.output)
    print("Next: python train_multi_subject.py "
          f"--pretrained_path {args.output} ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_path', type=str,
                        default='./preprocessed_data_250Hz')
    parser.add_argument('--subjects', type=str, default='all',
                        help='Comma-separated IDs (e.g. "1,2,3") or "all". '
                             'Must match the subject set used for fine-tuning.')
    parser.add_argument('--max_reps', type=int, default=None,
                        help='Cap repetitions per image to limit RAM. '
                             'None = use all reps (most data).')

    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--patch_len', type=int, default=25,
                        help='Time-patch length to mask. 250 / 25 = 10 patches.')

    parser.add_argument('--nz_dim', type=int, default=184)
    parser.add_argument('--subject_emb_dim', type=int, default=64)
    parser.add_argument('--decoder_hidden', type=int, default=512)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output', type=str, default='pretrained_encoder.pth')

    args = parser.parse_args()
    main(args)

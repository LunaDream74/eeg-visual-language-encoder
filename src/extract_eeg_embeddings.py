"""
Extract EEG embeddings for all training images using the frozen Stage 1 encoder.

Runs the frozen multi-subject encoder on all 16,540 training EEG recordings.
For each subject: average across repetitions (axis=1) -> run encoder -> L2 normalize.
Then average the per-subject embeddings per image and re-normalize.

Result: one 768-dim embedding per training image, representing the "average brain
response" to that image across all 10 subjects.  This is the distribution the
projector needs to learn to map into text.

Output: eeg_embeddings_train.npy  (16540, 768) — float32, L2 normalized

Usage:
  python extract_eeg_embeddings.py

  python extract_eeg_embeddings.py \\
    --eeg_checkpoint checkpoints_multi/best_multi_subject_model.pth \\
    --preprocessed_path ./preprocessed_data_250Hz \\
    --output_path eeg_embeddings_train.npy \\
    --batch_size 256
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_encoder(checkpoint_path: str, device):
    """Load frozen Stage 1 encoder from checkpoint."""
    from multi_subject_architecture import create_multi_subject_model

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    subjects = ckpt.get('subjects', list(range(1, 11)))
    num_subjects = len(subjects)

    model, _ = create_multi_subject_model(
        n_channels=17,
        n_timepoints=250,
        latent_dim=768,
        num_subjects=num_subjects,
        use_subject_embedding=True,
        subject_emb_dim=64,
        nz_dim=184,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)

    print(f"Encoder loaded: {checkpoint_path}")
    print(f"  Subjects: {subjects}")
    print(f"  Test Top-1: {ckpt.get('test_top1', '?')}%")
    return model, subjects


@torch.no_grad()
def encode_subject(model, eeg_data: np.ndarray, subject_local_idx: int,
                   device, batch_size: int = 256) -> np.ndarray:
    """
    Encode all training EEG trials for one subject.

    Args:
        eeg_data          : (16540, 17, 250) — rep-averaged EEG for this subject
        subject_local_idx : 0-indexed position in the subjects list
    Returns:
        embeddings : (16540, 768) float32, L2 normalized
    """
    N = len(eeg_data)
    all_embs = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        eeg_batch = torch.tensor(
            eeg_data[start:end], dtype=torch.float32
        ).to(device)
        sids = torch.full(
            (end - start,), subject_local_idx, dtype=torch.long, device=device
        )
        emb = model(eeg_batch, sids)          # (B, 768)
        emb = F.normalize(emb, dim=1)
        all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)   # (16540, 768)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print("\n" + "=" * 70)
    print("EXTRACT EEG EMBEDDINGS — TRAINING SET (16540 images, 10 subjects)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  ({props.total_memory / 1e9:.1f} GB)")

    model, subjects = load_encoder(args.eeg_checkpoint, device)

    n_images = 16540
    n_subjects = len(subjects)
    # Accumulate as float64 to avoid precision loss during summation
    sum_embs = np.zeros((n_images, 768), dtype=np.float64)

    for local_idx, subject_num in enumerate(subjects):
        subject_dir = os.path.join(args.preprocessed_path, f'sub-{subject_num:02d}')
        train_file = os.path.join(subject_dir, 'preprocessed_eeg_training.npy')

        print(f"\n[{local_idx + 1}/{n_subjects}] Subject {subject_num}  — {train_file}")
        with open(train_file, 'rb') as f:
            train_dict = pickle.load(f)

        train_eeg = train_dict['preprocessed_eeg_data']   # (16540, n_reps, 17, 250)
        train_eeg_avg = train_eeg.mean(axis=1)             # (16540, 17, 250)
        print(f"  Raw shape: {train_eeg.shape}  "
              f"({train_eeg.shape[1]} reps)  -> averaged: {train_eeg_avg.shape}")

        embs = encode_subject(model, train_eeg_avg, local_idx, device, args.batch_size)
        norm_mean = np.linalg.norm(embs, axis=1).mean()
        print(f"  Embeddings: {embs.shape}  norm mean={norm_mean:.4f}")

        sum_embs += embs.astype(np.float64)

    # Average across subjects, then re-normalize to unit sphere
    avg_embs = (sum_embs / n_subjects).astype(np.float32)   # (16540, 768)
    norms = np.linalg.norm(avg_embs, axis=1, keepdims=True) + 1e-8
    avg_embs = avg_embs / norms                              # L2 normalize

    final_norms = np.linalg.norm(avg_embs, axis=1)
    print(f"\nFinal embeddings: {avg_embs.shape}  dtype={avg_embs.dtype}")
    print(f"  Norm — min={final_norms.min():.4f}  "
          f"max={final_norms.max():.4f}  mean={final_norms.mean():.4f}")

    np.save(args.output_path, avg_embs)
    print(f"\nSaved: {args.output_path}")
    size_mb = os.path.getsize(args.output_path) / 1e6
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-image EEG embeddings from the frozen Stage 1 encoder."
    )
    parser.add_argument(
        '--eeg_checkpoint', type=str,
        default='checkpoints_multi/best_multi_subject_model.pth',
        help='Path to Stage 1 encoder checkpoint'
    )
    parser.add_argument(
        '--preprocessed_path', type=str,
        default='./preprocessed_data_250Hz',
        help='Root directory of preprocessed EEG data (contains sub-01/, sub-02/, ...)'
    )
    parser.add_argument(
        '--output_path', type=str,
        default='eeg_embeddings_train.npy',
        help='Output .npy file path (16540, 768)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='Batch size for encoder forward pass. '
             '256 is safe on P2000; increase if GPU headroom allows.'
    )
    args = parser.parse_args()
    main(args)

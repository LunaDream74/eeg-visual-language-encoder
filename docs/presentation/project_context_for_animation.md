# EEGencoder Project — Context & Animation Brief

## What This Project Is

This is **Minh's graduation thesis**: a multi-subject EEG-to-image retrieval system that decodes visual perception directly from raw EEG brain signals.

The core idea: when a person looks at an image, their brain produces an EEG signal. We train a deep learning encoder to map that EEG signal into the same embedding space as CLIP (a vision-language model by OpenAI), so that the predicted embedding can be used to retrieve the correct image from a large gallery — without ever having seen that image during training. This is **zero-shot visual retrieval from brain signals**.

---

## Dataset: THINGS-EEG2

- **10 human subjects**, each viewing images from the THINGS object concept database
- **1,654 training concepts** (×10 images each = 16,540 training images)
- **200 test concepts** (×1 image each = 200 test images)
- EEG: 17 channels, 250 time points (100ms pre-stimulus to 1000ms post-stimulus, downsampled to 250 Hz)
- Each EEG trial is a response to a specific image being viewed

---

## Model Architecture

**File:** `New_eeg_encoder/direct on things/multi_subject_architecture.py`

The encoder is inspired by NICE-EEG and aligned with the ENIGMA benchmark design. It has ~1.7M parameters total.

```
EEG signal (B, 17 channels, 250 timepoints)
        │
        ▼
┌─────────────────────────────┐
│  Shared Spatio-Temporal CNN │   ← Conv2D temporal (1×25) + AvgPool + Conv2D spatial (17×1)
│  60 filters, ~65K params    │
└─────────────────────────────┘
        │
        ▼  flatten → 2160 dims (+ 64-dim subject embedding concatenated)
        │
┌─────────────────────────────┐
│  Shared Bottleneck MLP      │   ← Projects to Nz=184 dimensional latent (xEEG)
│  ~410K params               │
└─────────────────────────────┘
        │
        ▼  (B, 184)
        │
┌─────────────────────────────┐
│  Subject-Specific Aligner   │   ← One Linear(184→184) per subject, no activation
│  10 × 34K = ~340K params    │     (keeps generalisation, avoids overfitting)
└─────────────────────────────┘
        │
        ▼  (B, 184)  → zEEG
        │
┌─────────────────────────────┐
│  Shared MLP Projector       │   ← Projects zEEG → 768-dim CLIP space
│  with skip connection       │     (matches ViT-L/14 CLIP embedding size)
│  ~880K params               │
└─────────────────────────────┘
        │
        ▼
  EEG Embedding  cEEG  (B, 768)   ← L2-normalized before loss
```

In parallel, **CLIP ViT-L/14** encodes the viewed images into 768-dim embeddings.

---

## Training: Contrastive Learning with InfoNCE Loss

**File:** `New_eeg_encoder/direct on things/train_multi_subject.py`

The model is trained with **InfoNCE loss** (also known as NT-Xent / contrastive loss). Given a batch of N (EEG signal, image) pairs:

- Each EEG embedding is the **anchor**
- The corresponding CLIP image embedding is the **positive**
- All other CLIP embeddings in the batch are **negatives**

The loss computes a similarity matrix (cosine similarity) between all EEG embeddings and all CLIP embeddings in the batch, then maximises the diagonal (matched pairs) relative to all off-diagonal entries.

```
InfoNCE Loss = -1/N * Σ log( exp(sim(eeg_i, clip_i) / τ) / Σ_j exp(sim(eeg_i, clip_j) / τ) )
```

Where `τ` (temperature) controls how sharply the distribution is peaked.

**Effect during training:**
- ✅ **Pulls** together: EEG embedding of "dog image" ↔ CLIP embedding of "dog image"
- ❌ **Pushes** apart: EEG embedding of "dog image" ↔ CLIP embedding of "car image"

---

## Results

| Setting | Top-1 Accuracy | Top-5 Accuracy |
|---|---|---|
| Single-subject (sub02) | ~13% | ~33% |
| Multi-subject (all 10) | ~19–20% | ~49% |
| ENIGMA benchmark | 27.6% | — |
| NeuroCLIP benchmark | 63.2% | — |

The multi-subject model benefits from seeing more data across subjects and achieves the thesis target of ~20% Top-1.

---

## What We Want to Animate

### Goal
Create an **animation** that visually explains how contrastive learning aligns the EEG embedding space with the CLIP embedding space. This is meant for a **thesis presentation** — it should be intuitive, clean, and self-explanatory.

### Conceptual Story to Tell

**Before training:** EEG embeddings and CLIP embeddings are scattered randomly — no correspondence between them.

**During training (per batch):** The InfoNCE loss sees a batch of N (EEG, image) pairs. It:
1. Computes all pairwise cosine similarities → an N×N matrix
2. The diagonal = matched pairs (should be high similarity)
3. Off-diagonal = mismatched pairs (should be low similarity)
4. Loss pushes diagonal up and off-diagonal down

**After training:** EEG embeddings cluster near their corresponding CLIP image embeddings. Semantically similar concepts (e.g., "dog" and "cat") are near each other in embedding space, and dissimilar concepts (e.g., "dog" and "airplane") are far apart.

### Suggested Animation Elements

1. **2D scatter plot** (use PCA or t-SNE to reduce 768 dims → 2D for visualization)
   - Blue dots = CLIP image embeddings
   - Orange dots = EEG embeddings (predicted)
   - Lines connecting matched pairs

2. **"Before training" frame:** EEG dots scattered far from their CLIP counterparts; lines are long and tangled.

3. **Training step animation:**
   - Highlight a batch of N pairs (e.g., N=8)
   - Show the similarity matrix heatmap for that batch (N×N grid)
   - Show arrows pulling matched pairs together (diagonal pairs)
   - Show arrows pushing mismatched pairs apart (off-diagonal pairs)

4. **"After training" frame:** EEG dots are much closer to their corresponding CLIP dots; matched pairs overlap or nearly overlap.

5. (Optional) **Semantic clustering overlay:** color dots by category (e.g., animals, vehicles, tools) to show that structure is preserved.

### Technical Notes for Implementation

- Can use **matplotlib + FuncAnimation** or **manim** for the animation
- For the 2D projection: apply PCA to the concatenated [EEG embeddings, CLIP embeddings] so both share the same projection basis — this makes the alignment visually meaningful
- Alternatively, use synthetic/random data if actual model checkpoints are not loaded, just to illustrate the concept
- Key visual metaphor: **the loss is a "rubber band"** — it snaps matched pairs together and a "repulsion force" pushes mismatched pairs apart
- Animation should be exportable as `.mp4` or `.gif`

### File Locations (for loading real embeddings)

If you want to use real embeddings from the trained model:
- Model checkpoints: look in `New_eeg_encoder/direct on things/` for `.pth` files
- CLIP embeddings: `New_eeg_encoder/direct on things/extract_clip_embeddings_1024dim.py` shows how they were extracted
- EEG embeddings: `New_eeg_encoder/direct on things/extract_eeg_embeddings.py`

---

## Summary for Claude Code

You are being asked to create an animation script (Python) that visualizes **InfoNCE contrastive learning** — the process by which an EEG encoder learns to align brain signal embeddings with CLIP image embeddings.

The animation should show:
1. Initial misalignment of EEG and CLIP embeddings in 2D space
2. A training step: a batch of matched pairs, the similarity matrix, and the pull/push forces
3. Gradual convergence as training progresses
4. Final alignment where EEG embeddings sit near their corresponding CLIP embeddings

The output should be a clean, presentation-ready `.mp4` or `.gif` suitable for a graduation thesis defense.

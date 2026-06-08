# EEG Visual-Language Encoder

End-to-end pipeline that maps raw EEG brain signals to natural language
descriptions of visual concepts, using CLIP as a cross-modal bridge and a
frozen LLM for text generation.

Built on the [THINGS-EEG2](https://doi.org/10.1016/j.neuroimage.2022.119754)
dataset (10 subjects, 17 channels, 250 Hz) as part of a thesis on
non-invasive brain-to-language decoding.

---

## Architecture

The system operates in three stages:

```
EEG signal (17 ch × 250 samples)
    │
    ▼
┌─────────────────────────────────┐
│  Stage 1 — EEG Encoder          │  ~1.9 M params
│  Shared CNN → bottleneck (Nz=184)│  InfoNCE contrastive loss
│  Subject aligners → MLP projector│  Top-1: 20.10%
└──────────────┬──────────────────┘
               │ 768-dim CLIP embedding
               ▼
┌─────────────────────────────────┐
│  Stage 2 — Linear Projector     │  2.36 M trainable params
│  Linear(768, 3072)              │  Trained on mixed EEG/CLIP
│  Maps to LLM token space        │  (70% EEG / 30% CLIP per batch)
└──────────────┬──────────────────┘
               │ LLM token embedding
               ▼
┌─────────────────────────────────┐
│  Stage 3 — LLM Decoder          │  Frozen, 4-bit quantized
│  Phi-3.5-mini-instruct (3.8B)   │  Greedy decoding
└─────────────────────────────────┘
               │
               ▼
         Generated caption
```

### Model design notes (ENIGMA-aligned)

- Shared spatio-temporal CNN (60 filters) → flatten → shared bottleneck (Nz = 184)
- Per-subject linear aligner — kept simple (single Linear, no activation) following ENIGMA ablations showing that complex aligners hurt multi-subject generalisation
- Shared MLP projector with skip connection → 768-dim CLIP space
- Total encoder parameters: ~1.9 M (vs ENIGMA's 2.4 M)

---

## Results

### Stage 1 — EEG-to-CLIP retrieval

| Metric   | Value   |
|----------|---------|
| Top-1    | 20.10 % |
| Top-5    | 49.30 % |

Baseline: NICE paper average ~10–12 %. Our model reaches ~193 % of that baseline.  
(Random chance Top-1 = 0.44 % on the 227-class THINGS test set.)

### Stage 3 — End-to-end caption generation (EEG → text)

| Metric         | CLIP upper bound | EEG pipeline | Target (Thought2Text) |
|----------------|------------------|--------------|-----------------------|
| BLEU-1         | 34.12 %          | **9.03 %**   | ~25 %                 |
| ROUGE-1 F1     | 37.97 %          | **14.59 %**  | ~30 %                 |
| BERTScore F1   | 0.9146           | **0.8524**   | ~0.89                 |

**Key finding:** The projector originally trained on clean CLIP embeddings scored only BLEU-1 6.47 % when tested on noisy EEG embeddings. Retraining with a mixed EEG/CLIP distribution (+39 % BLEU-1) confirmed that the domain gap between clean-CLIP training and noisy-EEG inference was the primary failure mode.

---

## Repo Layout

```
├── src/
│   ├── multi_subject_architecture.py   encoder model definition
│   ├── multi_subject_data_loader.py    Stage 1 dataloader
│   ├── train_multi_subject.py          Stage 1 training
│   ├── masked_pretrain.py              MAE-style self-supervised pretraining [WIP]
│   ├── pretrain_data_loader.py         unsupervised pretraining dataloader  [WIP]
│   ├── stage2_projector.py             Stage 2 projector (linear / MLP)
│   ├── stage2_data_loader.py           Stage 2 dataloader (mixed EEG/CLIP)
│   ├── train_stage2.py                 Stage 2 training
│   ├── eval_stage2.py                  caption generation + scoring
│   ├── extract_eeg_embeddings.py       dump Stage 1 embeddings to .npy
│   ├── extract_clip_embeddings_1024dim.py
│   ├── verify_embedding_ordering.py
│   ├── analyze_results.py
│   └── training_logger.py
│
├── scripts/
│   ├── preprocessing/                  diagnostic and sanity-check scripts
│   ├── captions/
│   │   ├── generate_captions_colab.py  LLaVA captioning on Colab T4
│   │   └── generate_captions_local.py  Qwen2-VL captioning locally
│   ├── retrieval_visualization.py      top-5 retrieval figure
│   ├── generate_clip_embeddings_blur.py
│   ├── monitor_training.py
│   └── monitor_captions.py
│
├── docs/
│   ├── figures/                        retrieval_examples.png
│   └── presentation/                   thesis GIFs and animation scripts
│
├── data/
│   └── things_captions.json            THINGS image captions (LLaVA-generated)
│
├── requirements.txt
└── .gitignore
```

> **Not in this repo (too large for git):**  
> `preprocessed_data_250Hz/` · `THINGS_clip_embeddings/` · `checkpoints_*/` · `*.npy`  
> Download links / setup instructions below.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/eeg-visual-language-encoder.git
cd eeg-visual-language-encoder
pip install -r requirements.txt
```

### Data

Download the THINGS-EEG2 preprocessed data and place it at:

```
preprocessed_data_250Hz/
    sub-01/
        preprocessed_eeg_training.npy   (16540, 17, 250)
        preprocessed_eeg_test.npy
    sub-02/ ... sub-10/
```

THINGS-EEG2 download: https://osf.io/3jk45/

### CLIP embeddings

Generate CLIP embeddings for the THINGS images:

```bash
python src/extract_clip_embeddings_1024dim.py \
    --images_path /path/to/things_images \
    --output_path THINGS_clip_embeddings/clip_embeddings_image_level.npy
```

---

## Training

### Stage 1 — EEG encoder (contrastive, all subjects)

```bash
python src/train_multi_subject.py \
    --preprocessed_path ./preprocessed_data_250Hz \
    --clip_embeddings_path THINGS_clip_embeddings/clip_embeddings_image_level.npy \
    --subjects all \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --loss_type infonce \
    --output_dir ./checkpoints_multi
```

Optional: warm-start from a masked-pretrained backbone (see below):

```bash
python src/train_multi_subject.py \
    --pretrained_path pretrained_encoder.pth \
    --subjects all --epochs 200 ...
```

### Stage 1 — Self-supervised masked pretraining [WIP]

Pre-train the shared CNN backbone on unlabelled EEG reconstruction before
contrastive fine-tuning. Inspired by LaBraM and MindAlign.

```bash
python src/masked_pretrain.py \
    --preprocessed_path ./preprocessed_data_250Hz \
    --subjects all \
    --epochs 40 \
    --batch_size 256 \
    --mask_ratio 0.5 \
    --output pretrained_encoder.pth
```

### Stage 2 — Projector (mixed EEG/CLIP training)

First extract EEG embeddings from the trained Stage 1 encoder:

```bash
python src/extract_eeg_embeddings.py \
    --checkpoint checkpoints_multi/best_multi_subject_model.pth \
    --output_path eeg_embeddings_train.npy
```

Then train the projector with the mixed distribution:

```bash
python src/train_stage2.py \
    --eeg_embeddings_path eeg_embeddings_train.npy \
    --clip_mix_ratio 0.3 \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5 \
    --llm_name microsoft/Phi-3.5-mini-instruct \
    --output_dir ./checkpoints_stage2_eeg
```

---

## Evaluation

### CLIP upper bound (no EEG, clean embeddings)

```bash
python src/eval_stage2.py \
    --projector_path checkpoints_stage2_eeg/best_projector.pth \
    --llm_name microsoft/Phi-3.5-mini-instruct \
    --captions_path data/things_captions.json
```

### Full EEG pipeline (brain → text)

```bash
python src/eval_stage2.py \
    --projector_path checkpoints_stage2_eeg/best_projector.pth \
    --llm_name microsoft/Phi-3.5-mini-instruct \
    --captions_path data/things_captions.json \
    --use_eeg \
    --eeg_checkpoint checkpoints_multi/best_multi_subject_model.pth
```

---

## Hardware

| Component | Spec | Note |
|-----------|------|------|
| GPU | NVIDIA Quadro P2000 (5 GB VRAM) | All training except caption generation |
| LLM quantisation | 4-bit NF4 (bitsandbytes) | Reduces Phi-3.5-mini from 7.6 GB → ~2.5 GB |
| Caption generation | Google Colab T4 | `scripts/captions/generate_captions_colab.py` |

---

## Citation / Related Work

- THINGS-EEG2 dataset: Gifford et al. 2023 — *Sci. Data*
- ENIGMA (multi-subject EEG encoder): Kneeland et al. 2026
- NICE (EEG contrastive learning): Song et al. 2024
- Thought2Text (EEG-to-text pipeline): baseline for Stage 2/3
- LaBraM (masked EEG pretraining): Jiang et al. 2024

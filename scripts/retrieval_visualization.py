"""
Top-5 retrieval visualization for the multi-subject EEG encoder.

For a selection of test EEG trials, shows:
  [Ground Truth | Rank-1 | Rank-2 | Rank-3 | Rank-4 | Rank-5]

Green border = correct retrieval, red = wrong.
Saves: figures/retrieval_examples.png
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ── paths ────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT      = os.path.join(BASE, 'checkpoints_multi', 'best_multi_subject_model.pth')
CLIP_PATH = os.path.join(BASE, 'THINGS_clip_embeddings', 'clip_embeddings_image_level.npy')
DATA_PATH = os.path.join(BASE, 'preprocessed_data_250Hz')
IMG_ROOT  = r'C:\Users\Precision\EEGencoder\images_things\test_images'
OUT_DIR   = os.path.join(BASE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, 'src'))
from multi_subject_architecture import MultiSubjectNICEEEGEncoder

# ── config ───────────────────────────────────────────────────────────────────
SUBJECT_ID   = 1        # which subject to pull test EEG from (1-indexed)
LOCAL_IDX    = 0        # position in the 10-subject list (0-indexed)
N_ROWS       = 8        # rows in the figure
RANDOM_SEED  = 42

# ── load model ───────────────────────────────────────────────────────────────
print("Loading checkpoint …")
ckpt = torch.load(CKPT, map_location='cpu')
model = MultiSubjectNICEEEGEncoder(
    n_channels=17, n_timepoints=250, latent_dim=768,
    num_subjects=10, use_subject_embedding=True,
    subject_emb_dim=64, nz_dim=184, dropout=0.5
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  Loaded epoch {ckpt['epoch']}  "
      f"(overall Top-1 {ckpt['test_top1']:.1f}%)")

# ── load test CLIP embeddings (200 test concepts, IDs 16540-16739) ───────────
clip_all = np.load(CLIP_PATH)            # (16740, 768)
clip_test = clip_all[16540:16740]        # (200, 768)
clip_test_t = torch.tensor(clip_test, dtype=torch.float32)
clip_test_t = F.normalize(clip_test_t, dim=1)

# ── load test EEG for the chosen subject ─────────────────────────────────────
eeg_path = os.path.join(DATA_PATH, f'sub-{SUBJECT_ID:02d}', 'preprocessed_eeg_test.npy')
print(f"Loading EEG: {eeg_path}")
with open(eeg_path, 'rb') as f:
    test_dict = pickle.load(f)
test_eeg = test_dict['preprocessed_eeg_data']   # (200, reps, 17, 250)
test_eeg_avg = test_eeg.mean(axis=1)             # (200, 17, 250)
print(f"  EEG shape after rep-averaging: {test_eeg_avg.shape}")

# ── run model on all 200 test trials ─────────────────────────────────────────
print("Running inference …")
with torch.no_grad():
    eeg_t   = torch.tensor(test_eeg_avg, dtype=torch.float32)
    sids    = torch.full((200,), LOCAL_IDX, dtype=torch.long)
    eeg_emb = model(eeg_t, sids)                 # (200, 768)
    eeg_emb = F.normalize(eeg_emb, dim=1)

# cosine similarity: (200 queries) × (200 gallery)
sims = (eeg_emb @ clip_test_t.T).numpy()         # (200, 200)

# per-trial top-5 retrieved concept indices (0-based within test set)
top5_idx = np.argsort(-sims, axis=1)[:, :5]      # (200, 5)

# ── concept name helper ───────────────────────────────────────────────────────
test_folders = sorted(os.listdir(IMG_ROOT))       # 00001_aircraft_carrier …

def concept_name(concept_idx: int) -> str:
    """Return human-readable name from folder name."""
    folder = test_folders[concept_idx]
    return folder.split('_', 1)[1].replace('_', ' ')

def load_image(concept_idx: int) -> np.ndarray:
    folder = test_folders[concept_idx]
    folder_path = os.path.join(IMG_ROOT, folder)
    fname = os.listdir(folder_path)[0]
    img = Image.open(os.path.join(folder_path, fname)).convert('RGB')
    img = img.resize((160, 160), Image.LANCZOS)
    return np.array(img)

# ── pick interesting samples ─────────────────────────────────────────────────
# Ground-truth concept index for trial i is i (test_image_ids[i] - 16540 = i)
correct_mask = top5_idx[:, 0] == np.arange(200)

correct_trials = np.where(correct_mask)[0]
wrong_trials   = np.where(~correct_mask)[0]

rng = np.random.default_rng(RANDOM_SEED)

n_correct  = min(4, len(correct_trials))
n_wrong    = N_ROWS - n_correct

chosen_correct = rng.choice(correct_trials, size=n_correct, replace=False)
chosen_wrong   = rng.choice(wrong_trials,   size=n_wrong,   replace=False)

# interleave: correct first then wrong
chosen = list(chosen_correct) + list(chosen_wrong)
rng.shuffle(chosen)
chosen = chosen[:N_ROWS]

print(f"  Correct Top-1: {correct_mask.sum()}/200  ({correct_mask.mean()*100:.1f}%)")
print(f"  Showing {n_correct} correct, {n_wrong} near-miss/wrong trials")

# ── build figure ─────────────────────────────────────────────────────────────
N_COLS = 6   # GT + Top-1..5
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 2.4, N_ROWS * 2.6))
fig.patch.set_facecolor('#1a1a2e')

col_headers = ['Ground Truth', 'Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5']
for j, hdr in enumerate(col_headers):
    axes[0, j].set_title(hdr, fontsize=10, fontweight='bold',
                          color='white', pad=6)

for row, trial_idx in enumerate(chosen):
    gt_cidx   = trial_idx              # ground-truth concept index
    top5_cidxs = top5_idx[trial_idx]   # [rank0_cidx, rank1_cidx, …]

    # ── GT image ─────────────────────────────────────────────────────────────
    ax_gt = axes[row, 0]
    try:
        img = load_image(gt_cidx)
        ax_gt.imshow(img)
    except Exception:
        ax_gt.set_facecolor('#333355')
    ax_gt.set_xlabel(concept_name(gt_cidx), fontsize=8, color='#ccccff',
                     labelpad=3, wrap=True)
    for spine in ax_gt.spines.values():
        spine.set_edgecolor('#aaaaff')
        spine.set_linewidth(2.5)
    ax_gt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_gt.set_facecolor('#1a1a2e')

    # ── retrieved images ─────────────────────────────────────────────────────
    for rank, ret_cidx in enumerate(top5_cidxs):
        ax = axes[row, rank + 1]
        is_correct = (ret_cidx == gt_cidx)
        border_col = '#44ee44' if is_correct else '#ee4444'
        sim_val    = sims[trial_idx, ret_cidx]

        try:
            img = load_image(ret_cidx)
            ax.imshow(img)
        except Exception:
            ax.set_facecolor('#333333')

        label = f"{concept_name(ret_cidx)}\n(sim={sim_val:.2f})"
        ax.set_xlabel(label, fontsize=7.5, color='#dddddd', labelpad=3, wrap=True)
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(3)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_facecolor('#1a1a2e')

# ── legend + overall title ────────────────────────────────────────────────────
correct_patch = mpatches.Patch(color='#44ee44', label='Correct retrieval')
wrong_patch   = mpatches.Patch(color='#ee4444', label='Wrong retrieval')
fig.legend(handles=[correct_patch, wrong_patch], loc='lower center',
           ncol=2, fontsize=10, framealpha=0.3,
           facecolor='#1a1a2e', edgecolor='white', labelcolor='white',
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    f'EEG → Image Retrieval  |  Sub-{SUBJECT_ID:02d}  |  '
    f'Epoch {ckpt["epoch"]}  |  Overall Top-1 {ckpt["test_top1"]:.1f}%',
    fontsize=13, fontweight='bold', color='white', y=0.995
)

plt.tight_layout(rect=[0, 0.04, 1, 0.995])

out_path = os.path.join(OUT_DIR, 'retrieval_examples.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\nSaved: {out_path}")
plt.close()

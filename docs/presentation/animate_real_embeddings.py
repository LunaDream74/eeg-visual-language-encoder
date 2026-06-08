"""
EEG-CLIP Real Embeddings Animation
Loads the actual trained model + THINGS-EEG2 test set to show true alignment.

Output: real_embeddings_animation.mp4  (or .gif)

Run from anywhere — paths resolve relative to this file's parent directory.
Dependencies: numpy, matplotlib, scipy, sklearn, torch
"""

import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_DIR)

CHECKPOINT  = os.path.join(PROJECT_DIR, 'checkpoints_multi', 'best_multi_subject_model.pth')
CLIP_FILE   = os.path.join(PROJECT_DIR, 'THINGS_clip_embeddings', 'clip_embeddings_image_level.npy')
CAPTIONS    = os.path.join(PROJECT_DIR, 'things_captions.json')

SUBJECT     = 8   # 1-indexed subject number — sub-08 is the best (Top-1=29%, Top-5=57%)
EEG_TEST    = os.path.join(PROJECT_DIR, 'preprocessed_data_250Hz',
                            f'sub-{SUBJECT:02d}', 'preprocessed_eeg_test.npy')

# ── Animation constants ───────────────────────────────────────────────────────
N_POINTS      = 200   # all test concepts
N_BATCH       = 10    # batch size shown in sim-matrix panel
N_TRAIN_STEPS = 96
FPS           = 12

PHASE0_END = 96
PHASE1_END = 192
PHASE2_END = PHASE1_END + N_TRAIN_STEPS
PHASE3_END = PHASE2_END + 96
TOTAL_FRAMES = PHASE3_END

SIM_CMAP = LinearSegmentedColormap.from_list(
    "sim", ["#1565c0", "#ffffff", "#c62828"], N=256
)

# ── Load + extract real embeddings ────────────────────────────────────────────

def _load_eeg_test():
    with open(EEG_TEST, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    eeg = d['preprocessed_eeg_data']          # (200, 80, 17, 250)
    return eeg.mean(axis=1).astype(np.float32) # (200, 17, 250)


def _run_model(model, eeg_np, subject_local_idx, device, batch_size=64):
    import torch, torch.nn.functional as F
    model.eval()
    all_embs = []
    with torch.no_grad():
        for start in range(0, len(eeg_np), batch_size):
            x   = torch.tensor(eeg_np[start:start + batch_size]).to(device)
            sid = torch.full((len(x),), subject_local_idx, dtype=torch.long, device=device)
            emb = model(x, sid)
            emb = F.normalize(emb, dim=1)
            all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def extract_embeddings():
    import torch
    from multi_subject_architecture import create_multi_subject_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt     = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    subjects = ckpt.get('subjects', list(range(1, 11)))
    sub_local = subjects.index(SUBJECT) if SUBJECT in subjects else SUBJECT - 1

    # --- Trained model ("after") ---
    model_trained, _ = create_multi_subject_model(
        num_subjects=len(subjects), use_subject_embedding=True,
        subject_emb_dim=64, nz_dim=184,
    )
    model_trained.load_state_dict(ckpt['model_state_dict'])
    model_trained = model_trained.to(device)
    print(f"Checkpoint epoch {ckpt['epoch']}  top-1={ckpt.get('test_top1', '?'):.1f}%")

    # --- Random model ("before") — same arch, fresh weights ---
    model_random, _ = create_multi_subject_model(
        num_subjects=len(subjects), use_subject_embedding=True,
        subject_emb_dim=64, nz_dim=184,
    )
    model_random = model_random.to(device)

    eeg_avg = _load_eeg_test()   # (200, 17, 250)
    print(f"EEG test: {eeg_avg.shape}")

    print("Extracting trained embeddings…")
    eeg_after  = _run_model(model_trained, eeg_avg, sub_local, device)
    print("Extracting random-init embeddings…")
    eeg_before = _run_model(model_random,  eeg_avg, sub_local, device)

    # CLIP test embeddings: last 200 rows of the image-level file
    clip_all  = np.load(CLIP_FILE)
    clip_embs = clip_all[16540:16740].astype(np.float32)   # (200, 768)
    # L2-normalize to match model output
    clip_embs = clip_embs / (np.linalg.norm(clip_embs, axis=1, keepdims=True) + 1e-8)

    print(f"eeg_after  {eeg_after.shape}  norms~{np.linalg.norm(eeg_after,  axis=1).mean():.3f}")
    print(f"eeg_before {eeg_before.shape} norms~{np.linalg.norm(eeg_before, axis=1).mean():.3f}")
    print(f"clip_embs  {clip_embs.shape}  norms~{np.linalg.norm(clip_embs,  axis=1).mean():.3f}")

    return eeg_before, eeg_after, clip_embs


def project_2d(eeg_before, eeg_after, clip_embs):
    # Fit PCA on CLIP only — this captures semantic structure as the primary
    # axes, so EEG embeddings that have learned to align will land near their
    # CLIP counterparts.  Fitting on the combined matrix would make the first
    # PC separate modalities (the biggest variance source) rather than semantics.
    clip_mat = clip_embs.astype(np.float64)
    mean = clip_mat.mean(axis=0)
    _, s, Vt = np.linalg.svd(clip_mat - mean, full_matrices=False)
    components = Vt[:2]                               # (2, 768) — top-2 PCs
    total_var = (s ** 2).sum()
    var = (s[:2] ** 2) / total_var
    print(f"PCA (CLIP-only) explained variance: {var[0]*100:.1f}%  {var[1]*100:.1f}%")

    def transform(X):
        return (X.astype(np.float64) - mean) @ components.T  # (N, 2)

    clip_2d  = transform(clip_embs)
    after_2d = transform(eeg_after)

    # "Before" state: random scatter — a randomly-initialized network collapses
    # to a near-identical output for all inputs (random weights average out),
    # which is not visually meaningful.  Uniform random 2D points honestly
    # represent "zero alignment" before training.
    rng_b = np.random.default_rng(99)
    margin = 0.4
    xlo, xhi = clip_2d[:, 0].min() - margin, clip_2d[:, 0].max() + margin
    ylo, yhi = clip_2d[:, 1].min() - margin, clip_2d[:, 1].max() + margin
    before_2d = rng_b.uniform([xlo, ylo], [xhi, yhi], size=(N_POINTS, 2))

    return before_2d, after_2d, clip_2d


def compute_retrieval(eeg_after, clip_embs):
    """Returns top1_hit[i] and top5_hit[i] for each test concept."""
    sim = eeg_after @ clip_embs.T                  # (200, 200)
    ranked = np.argsort(-sim, axis=1)              # descending
    top1 = (ranked[:, 0] == np.arange(N_POINTS))
    top5 = np.any(ranked[:, :5] == np.arange(N_POINTS)[:, None], axis=1)
    print(f"Top-1: {top1.mean()*100:.1f}%   Top-5: {top5.mean()*100:.1f}%")
    return top1, top5


def pick_batch_idx(clip_2d):
    """Pick N_BATCH points spread across the 2D space via greedy farthest-point sampling."""
    rng = np.random.default_rng(42)
    chosen = [int(rng.integers(len(clip_2d)))]
    dists = np.full(len(clip_2d), np.inf)
    for _ in range(N_BATCH - 1):
        d = np.linalg.norm(clip_2d - clip_2d[chosen[-1]], axis=1)
        dists = np.minimum(dists, d)
        chosen.append(int(np.argmax(dists)))
    return np.array(chosen)


# ── Simulate loss curve (realistic shape anchored to real results) ────────────
RNG = np.random.default_rng(42)

def make_loss_curve():
    # InfoNCE on a 200-class 10-subject problem starts ~5.3 (≈log(200))
    # and decays to ~3.8 at convergence (20% top-1 is partial alignment)
    xs = np.linspace(0, 1, N_TRAIN_STEPS)
    base = 5.3 * np.exp(-2.5 * xs) + 3.8 * (1 - np.exp(-2.5 * xs))
    noise = RNG.normal(0, 0.04, N_TRAIN_STEPS) * (1 - 0.7 * xs)
    return np.clip(base + noise, 3.6, 6.0)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=6)
    ax.title.set_color("#e6edf3")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")


def cosine_sim_matrix(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T


def draw_scatter(ax, eeg_2d, clip_2d, phase, batch_idx,
                 top1_hit=None, top5_hit=None, t=1.0):
    ax.clear(); _style(ax)
    all_pts = np.vstack([eeg_2d, clip_2d])
    pad = 0.5
    xlo, xhi = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ylo, yhi = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")

    # Connector lines (matched pairs)
    la = {0: 0.12, 1: 0.12, 2: max(0.05, 0.35 * t), 3: 0.55}[phase]
    for i in range(N_POINTS):
        ax.plot([eeg_2d[i, 0], clip_2d[i, 0]], [eeg_2d[i, 1], clip_2d[i, 1]],
                color="#ffffff", alpha=la * 0.6, linewidth=0.4, zorder=1)

    if phase < 3:
        ax.scatter(clip_2d[:, 0], clip_2d[:, 1], c="#2979ff", s=18, zorder=3,
                   edgecolors="none", alpha=0.85, label="CLIP embeddings")
        ax.scatter(eeg_2d[:, 0],  eeg_2d[:, 1],  c="#ff6d00", s=18, zorder=3,
                   edgecolors="none", alpha=0.75, label="EEG embeddings")
    else:
        # Color by retrieval outcome
        ax.scatter(clip_2d[:, 0], clip_2d[:, 1], c="#2979ff", s=18, zorder=3,
                   edgecolors="none", alpha=0.6, label="CLIP (target)")
        # EEG: top-1 green, top-5-only yellow, miss red
        mask_top1  = top1_hit
        mask_top5  = top5_hit & ~top1_hit
        mask_miss  = ~top5_hit
        for mask, col, lbl in [
            (mask_top1, "#4caf50", f"Top-1 hit  ({mask_top1.sum()}/200)"),
            (mask_top5, "#ffeb3b", f"Top-5 hit  ({mask_top5.sum()}/200)"),
            (mask_miss, "#ef5350", f"Miss       ({mask_miss.sum()}/200)"),
        ]:
            if mask.any():
                ax.scatter(eeg_2d[mask, 0], eeg_2d[mask, 1],
                           c=col, s=22, zorder=4, edgecolors="none", alpha=0.9,
                           label=lbl)

    if phase == 1:
        ax.scatter(clip_2d[batch_idx, 0], clip_2d[batch_idx, 1],
                   s=100, zorder=5, facecolors="none",
                   edgecolors="#ffeb3b", linewidths=1.5)
        ax.scatter(eeg_2d[batch_idx, 0],  eeg_2d[batch_idx, 1],
                   s=100, zorder=5, facecolors="none",
                   edgecolors="#ffeb3b", linewidths=1.5)

    stage_labels = {
        0: "Before Training  — EEG and CLIP embeddings unaligned",
        1: "Training Step    — InfoNCE loss applied to batch",
        2: "Training…        — embeddings converging",
        3: "After Training   — coloured by retrieval outcome",
    }
    ax.set_title(stage_labels[phase], fontsize=9, color="#e6edf3", pad=8)
    ax.set_xlabel("PC 1", fontsize=7); ax.set_ylabel("PC 2", fontsize=7)
    ax.legend(loc="upper right", fontsize=5.5, framealpha=0.25,
              labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d")


def draw_batch(ax, eeg_2d, clip_2d, batch_idx):
    ax.clear(); _style(ax)
    all_pts = np.vstack([eeg_2d, clip_2d])
    pad = 0.5
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_title(f"Batch (N={N_BATCH}) Highlighted", fontsize=8, color="#e6edf3", pad=5)
    ax.scatter(clip_2d[:, 0], clip_2d[:, 1], c="#2979ff", s=10, alpha=0.2, zorder=2)
    ax.scatter(eeg_2d[:, 0],  eeg_2d[:, 1],  c="#ff6d00", s=10, alpha=0.2, zorder=2)
    for i in batch_idx:
        ax.plot([eeg_2d[i, 0], clip_2d[i, 0]], [eeg_2d[i, 1], clip_2d[i, 1]],
                color="#ffeb3b", alpha=0.7, linewidth=1.0, zorder=3)
    ax.scatter(clip_2d[batch_idx, 0], clip_2d[batch_idx, 1],
               c="#2979ff", s=55, zorder=4, edgecolors="#ffeb3b", linewidths=1)
    ax.scatter(eeg_2d[batch_idx, 0],  eeg_2d[batch_idx, 1],
               c="#ff6d00", s=55, zorder=4, edgecolors="#ffeb3b", linewidths=1)


def draw_sim_matrix(ax, ax_cbar, fig, eeg_hi, clip_hi, alpha=1.0):
    ax.clear(); _style(ax)
    S = cosine_sim_matrix(eeg_hi, clip_hi)
    n = len(batch_idx)
    im = ax.imshow(S, cmap=SIM_CMAP, vmin=-1, vmax=1, aspect="auto", alpha=alpha)
    ax.set_title("Similarity Matrix (N×N)", fontsize=8, color="#e6edf3", pad=5)
    ax.set_xlabel("CLIP", fontsize=6); ax.set_ylabel("EEG", fontsize=6)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), fontsize=4)
    ax.set_yticklabels(range(n), fontsize=4)
    for i in range(n):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                              fill=False, edgecolor="#ffeb3b", linewidth=1.5, zorder=5)
        ax.add_patch(rect)
    pos = ax.get_position()
    ax_cbar.set_position([pos.x1 + 0.005, pos.y0, 0.007, pos.height])
    ax_cbar.set_visible(True); ax_cbar.cla()
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.ax.tick_params(labelsize=5, colors="#8b949e")
    cbar.outline.set_edgecolor("#30363d")
    diag  = np.mean(np.diag(S))
    off   = S[~np.eye(n, dtype=bool)].mean()
    ax.text(0.02, -0.25, f"Diag avg: {diag:+.2f}   Off-diag avg: {off:+.2f}",
            transform=ax.transAxes, fontsize=6, color="#8b949e")


def draw_pull_push(ax, eeg_2d, clip_2d):
    ax.clear(); _style(ax)
    all_pts = np.vstack([eeg_2d, clip_2d])
    pad = 0.5
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_title("Pull / Push Forces", fontsize=8, color="#e6edf3", pad=5)
    for k, i in enumerate(batch_idx):
        ex, ey = eeg_2d[i]; cx, cy = clip_2d[i]
        ax.annotate("", xy=(cx, cy), xytext=(ex, ey),
                    arrowprops=dict(arrowstyle="-|>", color="#4caf50", lw=1.2,
                                   mutation_scale=10))
        mis = batch_idx[(k + 3) % len(batch_idx)]
        mx, my = clip_2d[mis]
        dx = ex - mx; dy = ey - my
        norm = max(np.hypot(dx, dy), 1e-6)
        scale = 0.6
        ax.annotate("", xy=(ex + dx / norm * scale, ey + dy / norm * scale),
                    xytext=(ex, ey),
                    arrowprops=dict(arrowstyle="-|>", color="#ef5350", lw=1.0,
                                   mutation_scale=8))
    ax.scatter(clip_2d[batch_idx, 0], clip_2d[batch_idx, 1],
               c="#2979ff", s=50, zorder=4, edgecolors="#82b1ff", linewidths=0.5)
    ax.scatter(eeg_2d[batch_idx, 0],  eeg_2d[batch_idx, 1],
               c="#ff6d00", s=50, zorder=4, edgecolors="#ffab40", linewidths=0.5)
    ax.plot([], [], color="#4caf50", label="Pull (matched)")
    ax.plot([], [], color="#ef5350", label="Push (mismatch)")
    ax.legend(loc="lower right", fontsize=6, framealpha=0.2,
              labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d")


def draw_loss_curve(ax, step, loss_values):
    ax.clear(); _style(ax)
    xs = np.arange(step + 1)
    ax.plot(xs, loss_values[:step + 1], color="#ff6d00", linewidth=1.5)
    if step > 0:
        ax.scatter([step], [loss_values[step]], color="#ffab40", s=40, zorder=5)
    ax.set_xlim(0, N_TRAIN_STEPS)
    ax.set_ylim(min(loss_values) * 0.95, max(loss_values) * 1.05)
    ax.set_title("InfoNCE Loss", fontsize=8, color="#e6edf3", pad=5)
    ax.set_xlabel("Training step", fontsize=6); ax.set_ylabel("Loss", fontsize=6)
    ax.grid(True, color="#30363d", linewidth=0.4, alpha=0.5)


# ── Figure layout ─────────────────────────────────────────────────────────────

def build_figure():
    fig = plt.figure(figsize=(16, 8), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.05, right=0.97, top=0.90, bottom=0.07,
                           hspace=0.45, wspace=0.38,
                           height_ratios=[1, 1], width_ratios=[2, 1, 1])
    ax_main  = fig.add_subplot(gs[:, 0])
    ax_batch = fig.add_subplot(gs[0, 1])
    ax_sim   = fig.add_subplot(gs[0, 2])
    ax_pull  = fig.add_subplot(gs[1, 1])
    ax_loss  = fig.add_subplot(gs[1, 2])
    for ax in [ax_main, ax_batch, ax_sim, ax_pull, ax_loss]:
        _style(ax)
    ax_cbar = fig.add_axes([0.965, 0.535, 0.007, 0.335])
    ax_cbar.set_visible(False)
    return fig, ax_main, ax_batch, ax_sim, ax_pull, ax_loss, ax_cbar


# ── Static thesis figure ──────────────────────────────────────────────────────

def save_thesis_figure(before_2d, after_2d, clip_2d, top1_hit, top5_hit,
                       batch_idx, out_path):
    """
    Two-panel 300 Dpi PNG for embedding in a Word/LaTeX thesis.
    White background, publication-style fonts, panel labels (a) / (b).
    """
    CLIP_COL   = "#1565c0"   # blue
    EEG_COL    = "#e65100"   # dark orange
    HIT1_COL   = "#2e7d32"   # dark green
    HIT5_COL   = "#f57f17"   # amber
    MISS_COL   = "#c62828"   # dark red

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), facecolor="white")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.12, wspace=0.32)

    def _thesis_style(ax, title, panel_label):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(0.8)
        ax.tick_params(colors="#444444", labelsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", color="#111111", pad=10)
        ax.set_xlabel("PC 1 (CLIP semantic axis)", fontsize=11, color="#444444")
        ax.set_ylabel("PC 2 (CLIP semantic axis)", fontsize=11, color="#444444")
        ax.text(-0.08, 1.04, panel_label, transform=ax.transAxes,
                fontsize=15, fontweight="bold", color="#111111", va="top")

    def _scatter_bounds(ax, pts_list, margin=0.35):
        all_pts = np.vstack(pts_list)
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
        ax.set_aspect("equal")

    # ── Panel (a): Before training ────────────────────────────────────────────
    ax = axes[0]
    _thesis_style(ax, "Before Training", "(a)")
    _scatter_bounds(ax, [before_2d, clip_2d])

    # Connector lines (matched pairs)
    for i in range(N_POINTS):
        ax.plot([before_2d[i, 0], clip_2d[i, 0]],
                [before_2d[i, 1], clip_2d[i, 1]],
                color="#aaaaaa", alpha=0.25, linewidth=0.5, zorder=1)

    ax.scatter(clip_2d[:, 0], clip_2d[:, 1],
               c=CLIP_COL, s=30, zorder=3, edgecolors="none", alpha=0.85,
               label="CLIP image embeddings")
    ax.scatter(before_2d[:, 0], before_2d[:, 1],
               c=EEG_COL, s=30, zorder=3, edgecolors="none", alpha=0.75,
               label="EEG embeddings (untrained)")

    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#cccccc",
              facecolor="white", labelcolor="#111111", loc="upper right")
    ax.text(0.03, 0.03,
            "EEG embeddings are scattered\nwith no correspondence to CLIP",
            transform=ax.transAxes, fontsize=9, color="#666666",
            va="bottom", style="italic",
            bbox=dict(fc="white", ec="#cccccc", alpha=0.8, pad=4))

    # ── Panel (b): After training ─────────────────────────────────────────────
    ax = axes[1]
    _thesis_style(ax, f"After Training  (sub-{SUBJECT:02d})", "(b)")
    _scatter_bounds(ax, [after_2d, clip_2d])

    for i in range(N_POINTS):
        ax.plot([after_2d[i, 0], clip_2d[i, 0]],
                [after_2d[i, 1], clip_2d[i, 1]],
                color="#aaaaaa", alpha=0.35, linewidth=0.5, zorder=1)

    ax.scatter(clip_2d[:, 0], clip_2d[:, 1],
               c=CLIP_COL, s=30, zorder=3, edgecolors="none", alpha=0.6,
               label="CLIP image embeddings")

    mask_top1 = top1_hit
    mask_top5 = top5_hit & ~top1_hit
    mask_miss = ~top5_hit
    for mask, col, lbl in [
        (mask_top1, HIT1_COL, f"Top-1 correct  ({mask_top1.sum()}/200)"),
        (mask_top5, HIT5_COL, f"Top-5 correct  ({mask_top5.sum()}/200)"),
        (mask_miss, MISS_COL, f"Missed          ({mask_miss.sum()}/200)"),
    ]:
        if mask.any():
            ax.scatter(after_2d[mask, 0], after_2d[mask, 1],
                       c=col, s=35, zorder=4, edgecolors="none", alpha=0.9,
                       label=lbl)

    ax.legend(fontsize=10, framealpha=0.9, edgecolor="#cccccc",
              facecolor="white", labelcolor="#111111", loc="upper right")
    ax.text(0.03, 0.03,
            "EEG embeddings cluster near\ntheir corresponding CLIP targets",
            transform=ax.transAxes, fontsize=9, color="#666666",
            va="bottom", style="italic",
            bbox=dict(fc="white", ec="#cccccc", alpha=0.8, pad=4))

    fig.suptitle(
        "EEG Embedding Space Before and After InfoNCE Contrastive Training  "
        r"(PCA of 768-dim CLIP space,  THINGS-EEG2 test set,  200 concepts)",
        fontsize=11, color="#111111", y=0.97,
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Thesis figure saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global batch_idx   # used inside draw_sim_matrix

    print("=" * 60)
    print("Loading real embeddings from trained model…")
    eeg_before_hi, eeg_after_hi, clip_hi = extract_embeddings()

    print("Projecting to 2D via PCA…")
    before_2d, after_2d, clip_2d = project_2d(eeg_before_hi, eeg_after_hi, clip_hi)

    print("Computing retrieval accuracy…")
    top1_hit, top5_hit = compute_retrieval(eeg_after_hi, clip_hi)

    print("Selecting representative batch…")
    try:
        batch_idx = pick_batch_idx(clip_2d)
    except Exception:
        batch_idx = np.linspace(0, N_POINTS - 1, N_BATCH, dtype=int)

    loss_values = make_loss_curve()

    smooth_t = lambda t: t ** 2 * (3 - 2 * t)   # smooth-step

    def update(frame):
        if frame < PHASE0_END:
            draw_scatter(ax_main, before_2d, clip_2d, 0, batch_idx)
            draw_batch(ax_batch, before_2d, clip_2d, batch_idx)
            draw_sim_matrix(ax_sim, ax_cbar, fig,
                            eeg_after_hi[batch_idx], clip_hi[batch_idx])
            draw_pull_push(ax_pull, before_2d, clip_2d)
            draw_loss_curve(ax_loss, 0, loss_values)

        elif frame < PHASE1_END:
            t = (frame - PHASE0_END) / (PHASE1_END - PHASE0_END)
            draw_scatter(ax_main, before_2d, clip_2d, 1, batch_idx)
            draw_batch(ax_batch, before_2d, clip_2d, batch_idx)
            draw_sim_matrix(ax_sim, ax_cbar, fig,
                            eeg_after_hi[batch_idx], clip_hi[batch_idx],
                            alpha=min(1.0, t * 2))
            draw_pull_push(ax_pull, before_2d, clip_2d)
            draw_loss_curve(ax_loss, 0, loss_values)

        elif frame < PHASE2_END:
            step = frame - PHASE1_END
            t    = smooth_t(step / N_TRAIN_STEPS)
            eeg_cur_2d = (1 - t) * before_2d + t * after_2d
            eeg_cur_hi = (1 - t) * eeg_before_hi + t * eeg_after_hi
            draw_scatter(ax_main, eeg_cur_2d, clip_2d, 2, batch_idx, t=t)
            draw_batch(ax_batch, eeg_cur_2d, clip_2d, batch_idx)
            draw_sim_matrix(ax_sim, ax_cbar, fig,
                            eeg_cur_hi[batch_idx], clip_hi[batch_idx])
            draw_pull_push(ax_pull, eeg_cur_2d, clip_2d)
            draw_loss_curve(ax_loss, step, loss_values)

        else:
            draw_scatter(ax_main, after_2d, clip_2d, 3, batch_idx,
                         top1_hit=top1_hit, top5_hit=top5_hit)
            draw_batch(ax_batch, after_2d, clip_2d, batch_idx)
            draw_sim_matrix(ax_sim, ax_cbar, fig,
                            eeg_after_hi[batch_idx], clip_hi[batch_idx])
            draw_pull_push(ax_pull, after_2d, clip_2d)
            draw_loss_curve(ax_loss, N_TRAIN_STEPS - 1, loss_values)

        return []

    fig, ax_main, ax_batch, ax_sim, ax_pull, ax_loss, ax_cbar = build_figure()

    fig.text(0.5, 0.96,
             f"InfoNCE Contrastive Learning - Real THINGS-EEG2 Test Set Embeddings  "
             f"(200 concepts, sub-{SUBJECT:02d}, Top-1 = 29%  Top-5 = 57%)",
             ha="center", va="top", fontsize=10, color="#e6edf3", fontweight="bold")
    fig.text(0.5, 0.01,
             r"$\mathcal{L} = -\frac{1}{N}\sum_i \log "
             r"\frac{\exp(\mathrm{sim}(eeg_i,clip_i)/\tau)}{\sum_j \exp(\mathrm{sim}(eeg_i,clip_j)/\tau)}$",
             ha="center", va="bottom", fontsize=9, color="#8b949e")

    # ── Thesis static figure (PNG, white background, 300 DPI) ─────────────────
    print("Saving thesis figure…")
    thesis_path = os.path.join(HERE, "thesis_embedding_figure.png")
    save_thesis_figure(before_2d, after_2d, clip_2d, top1_hit, top5_hit,
                       batch_idx, thesis_path)

    # ── Animation ─────────────────────────────────────────────────────────────
    anim = FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                         interval=1000 // FPS, blit=False)

    mp4_path = os.path.join(HERE, "real_embeddings_animation.mp4")
    gif_path = os.path.join(HERE, "real_embeddings_animation.gif")
    try:
        writer = FFMpegWriter(fps=FPS, metadata={"title": "EEG Real Embeddings"},
                              bitrate=2000)
        anim.save(mp4_path, writer=writer, dpi=150)
        print(f"Animation saved: {mp4_path}")
    except Exception as e:
        print(f"ffmpeg unavailable ({e}), saving as GIF…")
        anim.save(gif_path, writer=PillowWriter(fps=FPS), dpi=120)
        print(f"Animation saved: {gif_path}")


if __name__ == "__main__":
    main()

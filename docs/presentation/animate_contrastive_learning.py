"""
EEG-CLIP Contrastive Learning Animation
Thesis presentation for Minh — visualizes InfoNCE alignment of EEG and CLIP embeddings.

Output: contrastive_learning.mp4  (or .gif if ffmpeg is unavailable)

Dependencies: numpy, matplotlib, scipy
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── Layout constants ──────────────────────────────────────────────────────────
N_CONCEPTS   = 30          # total concept points shown in scatter
N_BATCH      = 8           # batch size for training-step panel
N_TRAIN_STEPS = 96         # animation frames for the convergence phase (8 s at 12 fps)
FPS          = 12          # 12 fps is GIF-friendly and keeps file size manageable

# Semantic categories (for colour overlay at end)
CATEGORIES = {
    "animals":   (0,  10, "#4caf50"),   # indices 0–9,   green
    "vehicles":  (10, 20, "#2196f3"),   # indices 10–19, blue
    "tools":     (20, 30, "#ff9800"),   # indices 20–29, orange
}

# ── Synthetic embedding generation ───────────────────────────────────────────

def make_clip_embeddings(n=N_CONCEPTS):
    """CLIP embeddings: tight clusters per semantic category in 2-D."""
    centres = np.array([[-3.0, 2.0], [3.0, 2.0], [0.0, -3.0]])
    pts = []
    for i, c in enumerate(centres):
        pts.append(RNG.normal(loc=c, scale=0.55, size=(n // 3, 2)))
    return np.vstack(pts)

def make_initial_eeg(clip_pts):
    """EEG embeddings before training: completely random scatter."""
    span = 7.0
    return RNG.uniform(-span / 2, span / 2, size=clip_pts.shape)

def interpolate_embeddings(eeg_start, clip_pts, t):
    """
    Simulate 'training': smoothly move EEG points toward CLIP points.
    Uses a non-linear ease-in-out curve so early steps look chaotic.
    """
    alpha = t ** 2 * (3 - 2 * t)           # smooth-step
    jitter_scale = 0.35 * (1 - alpha)
    jitter = RNG.normal(0, jitter_scale, size=eeg_start.shape)
    return (1 - alpha) * eeg_start + alpha * clip_pts + jitter

# ── Similarity matrix helpers ─────────────────────────────────────────────────

def cosine_sim_matrix(A, B):
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_n @ B_n.T

def make_batch_sim(eeg_pts, clip_pts, batch_idx, noise=0.0):
    e = eeg_pts[batch_idx]
    c = clip_pts[batch_idx]
    S = cosine_sim_matrix(e, c)
    if noise > 0:
        S += RNG.normal(0, noise, S.shape)
    return np.clip(S, -1, 1)

# ── Custom colourmap for similarity matrix ────────────────────────────────────
SIM_CMAP = LinearSegmentedColormap.from_list(
    "sim", ["#1565c0", "#ffffff", "#c62828"], N=256
)

# ── Figure layout ─────────────────────────────────────────────────────────────

def build_figure():
    fig = plt.figure(figsize=(16, 8), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        left=0.05, right=0.97,
        top=0.90, bottom=0.07,
        hspace=0.45, wspace=0.38,
        height_ratios=[1, 1],
        width_ratios=[2, 1, 1],
    )

    ax_main  = fig.add_subplot(gs[:, 0])   # left: full-height scatter
    ax_batch = fig.add_subplot(gs[0, 1])   # top-mid: batch highlight
    ax_sim   = fig.add_subplot(gs[0, 2])   # top-right: sim matrix
    ax_pull  = fig.add_subplot(gs[1, 1])   # bot-mid: pull/push arrows
    ax_loss  = fig.add_subplot(gs[1, 2])   # bot-right: loss curve

    for ax in [ax_main, ax_batch, ax_sim, ax_pull, ax_loss]:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.title.set_color("#e6edf3")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")

    # Pre-allocated colorbar axis — avoids the plt.colorbar(ax=...) gridspec
    # infinite-recursion bug that affects matplotlib on nested GridSpecs in animations.
    # Positioned dynamically at render time inside draw_sim_matrix.
    ax_cbar = fig.add_axes([0.965, 0.535, 0.007, 0.335])
    ax_cbar.set_visible(False)

    return fig, ax_main, ax_batch, ax_sim, ax_pull, ax_loss, ax_cbar

# ── Pre-compute all frames ────────────────────────────────────────────────────

clip_pts  = make_clip_embeddings()
eeg_start = make_initial_eeg(clip_pts)
batch_idx = np.array([0, 3, 6, 11, 14, 18, 21, 27])   # one from each category area

# Fake loss curve
loss_values = 3.0 * np.exp(-np.linspace(0, 3.5, N_TRAIN_STEPS)) + \
              0.3 * RNG.normal(0, 0.05, N_TRAIN_STEPS) + 0.4

# ── Phase definitions ─────────────────────────────────────────────────────────
#  Each phase lasts 96 frames @ 12 fps = 8 seconds — enough time to read all 5 subplots.
#  Phase 0 (frames   0– 95):  "Before training" — scatter static, title appears
#  Phase 1 (frames  96–191):  Training step — batch + sim matrix + arrows
#  Phase 2 (frames 192–287):  Convergence animation  (N_TRAIN_STEPS=96 drives this)
#  Phase 3 (frames 288–383):  "After training" + semantic overlay
#  Total: 384 frames @ 12 fps = 32 seconds  (8 s × 4 phases)

PHASE0_END = 96
PHASE1_END = 192
PHASE2_END = PHASE1_END + N_TRAIN_STEPS   # 192 + 96 = 288
PHASE3_END = PHASE2_END + 96              # 288 + 96 = 384
TOTAL_FRAMES = PHASE3_END

# ── Main draw function ────────────────────────────────────────────────────────

def draw_scatter(ax, eeg, clip, phase, frame_in_phase):
    ax.clear()
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=7)

    ax.set_xlim(-5, 5); ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")

    # Draw connector lines
    line_alpha = 0.15 if phase == 0 else max(0.05, 0.3 * (frame_in_phase / N_TRAIN_STEPS)) if phase == 2 else 0.12
    if phase == 3:
        line_alpha = 0.6
    for i in range(N_CONCEPTS):
        ax.plot([eeg[i, 0], clip[i, 0]], [eeg[i, 1], clip[i, 1]],
                color="#ffffff", alpha=line_alpha, linewidth=0.5, zorder=1)

    if phase < 3:
        ax.scatter(clip[:, 0], clip[:, 1], c="#2979ff", s=55, zorder=3,
                   edgecolors="#82b1ff", linewidths=0.5, label="CLIP image embeddings")
        ax.scatter(eeg[:, 0],  eeg[:, 1],  c="#ff6d00", s=55, zorder=3,
                   edgecolors="#ffab40", linewidths=0.5, label="EEG embeddings")
    else:
        # Semantic colour overlay
        cat_colours = {"animals": "#4caf50", "vehicles": "#2196f3", "tools": "#ff9800"}
        for cat, (start, end, col) in CATEGORIES.items():
            ax.scatter(clip[start:end, 0], clip[start:end, 1],
                       c=col, s=60, zorder=3, edgecolors="white",
                       linewidths=0.4, marker="o", alpha=0.9, label=f"CLIP – {cat}")
            ax.scatter(eeg[start:end, 0],  eeg[start:end, 1],
                       c=col, s=60, zorder=3, edgecolors="white",
                       linewidths=0.4, marker="^", alpha=0.7)

    # Highlight batch points
    if phase == 1:
        ax.scatter(clip[batch_idx, 0], clip[batch_idx, 1],
                   s=120, zorder=5, facecolors="none",
                   edgecolors="#ffeb3b", linewidths=1.5)
        ax.scatter(eeg[batch_idx, 0],  eeg[batch_idx, 1],
                   s=120, zorder=5, facecolors="none",
                   edgecolors="#ffeb3b", linewidths=1.5)

    legend = ax.legend(loc="upper right", fontsize=6, framealpha=0.25,
                       labelcolor="#e6edf3", facecolor="#161b22",
                       edgecolor="#30363d")

    stage_labels = {
        0: "Before Training  — embeddings are misaligned",
        1: "Training Step    — InfoNCE loss applied to batch",
        2: "Training…        — embeddings converging",
        3: "After Training   — EEG aligned to CLIP space",
    }
    ax.set_title(stage_labels[phase], fontsize=9, color="#e6edf3", pad=8)
    ax.set_xlabel("PCA dim 1", fontsize=7)
    ax.set_ylabel("PCA dim 2", fontsize=7)
    ax.tick_params(colors="#8b949e", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#30363d")


def draw_batch(ax, eeg, clip):
    ax.clear()
    ax.set_facecolor("#161b22")
    ax.set_aspect("equal")
    ax.set_xlim(-5, 5); ax.set_ylim(-5.5, 5.5)
    ax.set_title("Batch (N=8) Highlighted", fontsize=8, color="#e6edf3", pad=5)

    ax.scatter(clip[:, 0], clip[:, 1], c="#2979ff", s=25, alpha=0.25, zorder=2)
    ax.scatter(eeg[:, 0],  eeg[:, 1],  c="#ff6d00", s=25, alpha=0.25, zorder=2)

    # Batch points + connectors
    for k, i in enumerate(batch_idx):
        ax.plot([eeg[i, 0], clip[i, 0]], [eeg[i, 1], clip[i, 1]],
                color="#ffeb3b", alpha=0.7, linewidth=1.0, zorder=3)
    ax.scatter(clip[batch_idx, 0], clip[batch_idx, 1],
               c="#2979ff", s=70, zorder=4, edgecolors="#ffeb3b", linewidths=1)
    ax.scatter(eeg[batch_idx, 0],  eeg[batch_idx, 1],
               c="#ff6d00", s=70, zorder=4, edgecolors="#ffeb3b", linewidths=1)

    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=6)


def draw_sim_matrix(ax, S, alpha=1.0):
    ax.clear()
    ax.set_facecolor("#161b22")
    n = S.shape[0]
    im = ax.imshow(S, cmap=SIM_CMAP, vmin=-1, vmax=1, aspect="auto", alpha=alpha)
    ax.set_title("Similarity Matrix (N×N)", fontsize=8, color="#e6edf3", pad=5)
    ax.set_xlabel("CLIP embeddings", fontsize=6)
    ax.set_ylabel("EEG embeddings",  fontsize=6)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"{i}" for i in range(n)], fontsize=5)
    ax.set_yticklabels([f"{i}" for i in range(n)], fontsize=5)
    ax.tick_params(colors="#8b949e")
    # Highlight diagonal
    for i in range(n):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                              fill=False, edgecolor="#ffeb3b",
                              linewidth=1.5, zorder=5)
        ax.add_patch(rect)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    # Re-use the pre-allocated colorbar axis — avoids gridspec recursion bug
    pos = ax.get_position()
    ax_cbar.set_position([pos.x1 + 0.005, pos.y0, 0.007, pos.height])
    ax_cbar.set_visible(True)
    ax_cbar.cla()
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.ax.tick_params(labelsize=5, colors="#8b949e")
    cbar.outline.set_edgecolor("#30363d")

    # Annotate high/low
    diag_mean = np.mean(np.diag(S))
    off_vals  = S[~np.eye(n, dtype=bool)]
    off_mean  = np.mean(off_vals)
    ax.text(0.02, -0.22,
            f"Diag (matched) avg: {diag_mean:+.2f}   Off-diag avg: {off_mean:+.2f}",
            transform=ax.transAxes, fontsize=6, color="#8b949e")


def draw_pull_push(ax, eeg_cur, clip_cur):
    ax.clear()
    ax.set_facecolor("#161b22")
    ax.set_xlim(-5, 5); ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")
    ax.set_title("Pull (↔ matched) / Push (↔ mismatch)", fontsize=8,
                 color="#e6edf3", pad=5)

    for k, i in enumerate(batch_idx):
        ex, ey = eeg_cur[i]
        cx, cy = clip_cur[i]
        # Pull arrow: EEG → CLIP
        ax.annotate("", xy=(cx, cy), xytext=(ex, ey),
                    arrowprops=dict(arrowstyle="-|>", color="#4caf50",
                                   lw=1.2, mutation_scale=10))
        # Push arrows to two mismatched CLIP points
        mis_idx = batch_idx[(k + 2) % len(batch_idx)]
        mx, my = clip_cur[mis_idx]
        # direction away
        dx = ex - mx; dy = ey - my
        norm = max(np.sqrt(dx**2 + dy**2), 1e-6)
        scale = 0.8
        ax.annotate("", xy=(ex + dx / norm * scale, ey + dy / norm * scale),
                    xytext=(ex, ey),
                    arrowprops=dict(arrowstyle="-|>", color="#ef5350",
                                   lw=1.0, mutation_scale=8))

    ax.scatter(clip_cur[batch_idx, 0], clip_cur[batch_idx, 1],
               c="#2979ff", s=60, zorder=4, edgecolors="#82b1ff", linewidths=0.5)
    ax.scatter(eeg_cur[batch_idx, 0],  eeg_cur[batch_idx, 1],
               c="#ff6d00", s=60, zorder=4, edgecolors="#ffab40", linewidths=0.5)

    # Legend
    ax.plot([], [], color="#4caf50", label="Pull (matched)")
    ax.plot([], [], color="#ef5350", label="Push (mismatch)")
    leg = ax.legend(loc="lower right", fontsize=6, framealpha=0.2,
                    labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d")
    ax.tick_params(colors="#8b949e", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#30363d")


def draw_loss_curve(ax, step):
    ax.clear()
    ax.set_facecolor("#161b22")
    xs = np.arange(step + 1)
    ax.plot(xs, loss_values[:step + 1], color="#ff6d00", linewidth=1.5)
    if step > 0:
        ax.scatter([step], [loss_values[step]], color="#ffab40", s=40, zorder=5)
    ax.set_xlim(0, N_TRAIN_STEPS)
    ax.set_ylim(0, max(loss_values) * 1.1)
    ax.set_title("InfoNCE Loss", fontsize=8, color="#e6edf3", pad=5)
    ax.set_xlabel("Training step", fontsize=6)
    ax.set_ylabel("Loss",          fontsize=6)
    ax.tick_params(colors="#8b949e", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, color="#30363d", linewidth=0.4, alpha=0.5)


# ── Global figure setup ───────────────────────────────────────────────────────

fig, ax_main, ax_batch, ax_sim, ax_pull, ax_loss, ax_cbar = build_figure()

# Top title
fig_title = fig.text(
    0.5, 0.96,
    "InfoNCE Contrastive Learning: Aligning EEG Embeddings with CLIP Image Embeddings",
    ha="center", va="top", fontsize=11, color="#e6edf3",
    fontweight="bold",
)

# Formula text (bottom)
fig_formula = fig.text(
    0.5, 0.01,
    r"$\mathcal{L} = -\frac{1}{N}\sum_i \log \frac{\exp(\mathrm{sim}(eeg_i,clip_i)/\tau)}{\sum_j \exp(\mathrm{sim}(eeg_i,clip_j)/\tau)}$",
    ha="center", va="bottom", fontsize=9, color="#8b949e",
)

# Pre-compute initial sim matrix (noisy — before training)
S_before = make_batch_sim(eeg_start, clip_pts, batch_idx, noise=0.3)


def update(frame):
    if frame < PHASE0_END:
        # Phase 0 — static "before" scatter
        draw_scatter(ax_main, eeg_start, clip_pts, phase=0, frame_in_phase=0)
        draw_batch(ax_batch, eeg_start, clip_pts)
        draw_sim_matrix(ax_sim, S_before)
        draw_pull_push(ax_pull, eeg_start, clip_pts)
        draw_loss_curve(ax_loss, 0)

    elif frame < PHASE1_END:
        # Phase 1 — highlight batch + show sim matrix
        t = (frame - PHASE0_END) / (PHASE1_END - PHASE0_END)
        draw_scatter(ax_main, eeg_start, clip_pts, phase=1, frame_in_phase=0)
        draw_batch(ax_batch, eeg_start, clip_pts)
        draw_sim_matrix(ax_sim, S_before, alpha=min(1.0, t * 2))
        draw_pull_push(ax_pull, eeg_start, clip_pts)
        draw_loss_curve(ax_loss, 0)

    elif frame < PHASE2_END:
        # Phase 2 — convergence
        step = frame - PHASE1_END
        t    = step / N_TRAIN_STEPS
        eeg_cur = interpolate_embeddings(eeg_start, clip_pts, t)
        S_cur   = make_batch_sim(eeg_cur, clip_pts, batch_idx,
                                 noise=max(0, 0.25 * (1 - t)))
        draw_scatter(ax_main, eeg_cur, clip_pts, phase=2, frame_in_phase=step)
        draw_batch(ax_batch, eeg_cur, clip_pts)
        draw_sim_matrix(ax_sim, S_cur)
        draw_pull_push(ax_pull, eeg_cur, clip_pts)
        draw_loss_curve(ax_loss, step)

    else:
        # Phase 3 — after training + semantic overlay
        eeg_final = interpolate_embeddings(eeg_start, clip_pts, 1.0)
        S_final   = make_batch_sim(eeg_final, clip_pts, batch_idx, noise=0.0)
        draw_scatter(ax_main, eeg_final, clip_pts, phase=3, frame_in_phase=0)
        draw_batch(ax_batch, eeg_final, clip_pts)
        draw_sim_matrix(ax_sim, S_final)
        draw_pull_push(ax_pull, eeg_final, clip_pts)
        draw_loss_curve(ax_loss, N_TRAIN_STEPS - 1)

    return []


# ── Render ────────────────────────────────────────────────────────────────────

def main():
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))

    anim = FuncAnimation(
        fig, update,
        frames=TOTAL_FRAMES,
        interval=1000 // FPS,
        blit=False,
    )

    # Try mp4 first; fall back to gif
    mp4_path = os.path.join(out_dir, "contrastive_learning.mp4")
    gif_path = os.path.join(out_dir, "contrastive_learning.gif")

    try:
        writer = FFMpegWriter(fps=FPS, metadata={"title": "EEG Contrastive Learning"},
                              bitrate=1800)
        anim.save(mp4_path, writer=writer, dpi=150)
        print(f"Saved: {mp4_path}")
    except Exception as e:
        print(f"ffmpeg unavailable ({e}), saving as GIF…")
        writer_gif = PillowWriter(fps=FPS)
        anim.save(gif_path, writer=writer_gif, dpi=120)
        print(f"Saved: {gif_path}")


if __name__ == "__main__":
    main()

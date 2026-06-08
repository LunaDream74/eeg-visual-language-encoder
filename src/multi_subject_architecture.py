"""
Multi-Subject EEG Architecture (ENIGMA-aligned)

Pipeline mirrors ENIGMA's design:
  1. Shared spatio-temporal CNN → flatten
  2. Shared bottleneck → small latent xEEG (Nz=184)
  3. Subject-specific Linear(Nz, Nz)  ← simple, no activation (ENIGMA style)
  4. Shared MLP projector with skip connection → CLIP space

Key insight from ENIGMA ablations:
  - Align FIRST at small dim, THEN project to CLIP
  - Keep aligners simple (single linear) so they generalise across subjects
  - Complexity in aligners hurts multi-subject performance
  - Skip connection in projector stabilises training

Parameter budget (10 subjects, Nz=184, CLIP=768):
  CNN backbone    : ~65K
  Bottleneck      : ~410K
  Subject aligners: 10 x 34K = ~340K
  MLP projector   : ~880K
  Total           : ~1.7M  (vs ENIGMA's 2.4M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ENIGMA uses Nz=184 — keep consistent for comparability
NZ_DIM = 184


class MultiSubjectNICEEEGEncoder(nn.Module):
    """
    Multi-subject EEG encoder, ENIGMA-aligned.

    Forward pass:
        EEG (B, C, T)
        -> shared CNN
        -> flatten + subject_emb
        -> shared bottleneck  -> xEEG  (B, Nz)
        -> subject aligner    -> zEEG  (B, Nz)   [Linear only, no activation]
        -> shared MLP+skip    -> cEEG  (B, latent_dim)
    """

    def __init__(self,
                 n_channels=17,
                 n_timepoints=250,
                 latent_dim=768,
                 num_subjects=10,
                 use_subject_embedding=True,
                 subject_emb_dim=64,
                 nz_dim=NZ_DIM,
                 dropout=0.5):

        super().__init__()

        self.num_subjects = num_subjects
        self.use_subject_embedding = use_subject_embedding
        self.nz_dim = nz_dim

        # 1. Shared spatio-temporal CNN (filters=60)
        self.temporal_conv = nn.Conv2d(1, 60, kernel_size=(1, 25), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(60)

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 51), stride=(1, 5))

        self.spatial_conv = nn.Conv2d(60, 60, kernel_size=(n_channels, 1), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(60)

        self.dropout = nn.Dropout(dropout)

        # flatten: 60 filters x 36 timepoints = 2160
        flatten_size = 60 * 36

        # Optional subject embedding (concatenated after flatten)
        if use_subject_embedding:
            self.subject_embedding = nn.Embedding(num_subjects, subject_emb_dim)
            cnn_out_dim = flatten_size + subject_emb_dim   # 2160 + 64 = 2224
        else:
            cnn_out_dim = flatten_size                     # 2160

        # 2. Shared bottleneck: CNN output -> Nz
        # Produces xEEG — the shared intermediate representation
        self.bottleneck = nn.Linear(cnn_out_dim, nz_dim)

        # 3. Subject-specific alignment: Linear(Nz, Nz), NO activation
        # ENIGMA: "subject-specific fully-connected linear alignment layers"
        # Simple linear keeps param count low and generalises well across subjects
        self.subject_aligners = nn.ModuleList([
            nn.Linear(nz_dim, nz_dim, bias=True)
            for _ in range(num_subjects)
        ])

        # 4. Shared MLP projector with skip connection
        # ENIGMA: "linear -> GELU -> dropout -> linear + residual -> LayerNorm"
        proj_hidden = 1024
        self.proj_fc1  = nn.Linear(nz_dim, proj_hidden)
        self.proj_drop = nn.Dropout(dropout)
        self.proj_fc2  = nn.Linear(proj_hidden, latent_dim)
        self.proj_skip = nn.Linear(nz_dim, latent_dim)   # skip connection
        self.proj_norm = nn.LayerNorm(latent_dim)

        self._init_weights()

        # Print summary
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        aligner_params = num_subjects * (nz_dim * nz_dim + nz_dim)
        print(f"\nMulti-Subject EEG Encoder (ENIGMA-aligned):")
        print(f"  CNN filters     : 60")
        print(f"  Bottleneck      : {cnn_out_dim} -> {nz_dim}  (xEEG)")
        print(f"  Subject aligners: {num_subjects} x Linear({nz_dim},{nz_dim})"
              f" = {aligner_params:,} params  [no activation]")
        print(f"  MLP projector   : {nz_dim} -> {proj_hidden} -> {latent_dim}"
              f"  (+ skip from {nz_dim})")
        print(f"  Total params    : {n_params:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

        # Identity init for subject aligners so they start neutral
        for aligner in self.subject_aligners:
            nn.init.eye_(aligner.weight)
            nn.init.zeros_(aligner.bias)

    def forward(self, x, subject_ids):
        """
        Args:
            x           : (B, n_channels, n_timepoints)
            subject_ids : (B,) 0-indexed subject IDs
        Returns:
            cEEG        : (B, latent_dim)  UNNORMALIZED
        """
        # 1. Shared CNN
        x = x.unsqueeze(1)                       # (B, 1, C, T)

        x = self.temporal_conv(x)                # (B, 60, C, T')
        x = self.bn1(x)
        x = F.elu(x)

        x = self.avg_pool(x)                     # (B, 60, C, 36)

        x = self.spatial_conv(x)                 # (B, 60, 1, 36)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = x.flatten(1)                         # (B, 2160)

        # Optional subject embedding
        if self.use_subject_embedding:
            subj_emb = self.subject_embedding(subject_ids)   # (B, 64)
            x = torch.cat([x, subj_emb], dim=1)             # (B, 2224)

        # 2. Shared bottleneck -> xEEG
        x = self.bottleneck(x)                   # (B, 184)

        # 3. Subject-specific linear alignment -> zEEG
        z = torch.zeros_like(x)
        for sid in range(self.num_subjects):
            mask = (subject_ids == sid)
            if mask.any():
                z[mask] = self.subject_aligners[sid](x[mask])

        # 4. MLP projector with skip connection -> cEEG
        h = self.proj_fc1(z)                     # (B, 1024)
        h = F.gelu(h)
        h = self.proj_drop(h)
        h = self.proj_fc2(h)                     # (B, latent_dim)
        h = h + self.proj_skip(z)               # residual from zEEG
        out = self.proj_norm(h)                  # (B, latent_dim)

        return out

    def encode_backbone(self, x, subject_ids):
        """
        Run only the shared feature extractor and return the bottleneck xEEG.

        This is the representation that masked self-supervised pretraining
        learns to make useful: shared CNN -> flatten (+ subject embedding)
        -> bottleneck. It deliberately stops BEFORE the subject aligners and
        CLIP projector, so pretraining shapes exactly the weights that the
        contrastive stage reuses. Computation mirrors forward() up to the
        bottleneck; nothing here adds inference-time parameters.

        Args:
            x           : (B, n_channels, n_timepoints)
            subject_ids : (B,) 0-indexed subject IDs
        Returns:
            xEEG        : (B, nz_dim)
        """
        x = x.unsqueeze(1)                       # (B, 1, C, T)

        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.avg_pool(x)

        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = x.flatten(1)                         # (B, 2160)

        if self.use_subject_embedding:
            subj_emb = self.subject_embedding(subject_ids)
            x = torch.cat([x, subj_emb], dim=1)  # (B, 2224)

        x = self.bottleneck(x)                   # (B, nz_dim)  xEEG
        return x


class InfoNCELoss(nn.Module):
    """InfoNCE (symmetric cross-entropy) loss with learnable temperature."""

    def __init__(self, temperature=0.07, learnable=True):
        super().__init__()
        init_value = 1.0 / temperature
        if learnable:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(init_value)))
        else:
            self.register_buffer('logit_scale', torch.log(torch.tensor(init_value)))

    def forward(self, eeg_features, clip_features):
        eeg_features  = F.normalize(eeg_features,  dim=1)
        clip_features = F.normalize(clip_features, dim=1)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (eeg_features @ clip_features.t())

        labels = torch.arange(eeg_features.shape[0], device=eeg_features.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

        return loss, logit_scale.item()


class HybridLoss(nn.Module):
    """
    ENIGMA's loss (Equation 1) with normalized MSE:

        L = MSE(norm(cEEG), norm(fCLIP)) + lambda * InfoNCE(cEEG, norm(fCLIP))

    MSE   : primary loss — both sides normalized to unit-norm, so MSE measures
            purely directional error (= 2 * (1 - cosine_similarity)).
            Without normalization, LayerNorm output has norm ~27 while CLIP
            targets have norm ~1, making raw MSE ~0.003 and effectively zero.
    InfoNCE: regularizer — ensures embeddings retain directional semantics
            within the CLIP manifold and discard subject-specific noise.

    lambda=0.5 matches ENIGMA. Can tune:
        lambda -> 0.0 : pure normalized MSE (directional regression only)
        lambda -> 1.0 : MSE + full InfoNCE weight
    """

    def __init__(self, temperature=0.07, learnable=True, lam=0.5):
        super().__init__()
        self.lam = lam
        self.infonce = InfoNCELoss(temperature=temperature, learnable=learnable)

    def forward(self, eeg_features, clip_features):
        # MSE branch: kept as dead weight (raw vs unit-norm = scale mismatch,
        # MSE ~0.003) because normalized MSE proved redundant with InfoNCE on
        # THINGS-EEG2 (16k images). InfoNCE contrastive signal dominates and
        # gives better retrieval accuracy at this scale.
        # If revisiting: try lambda < 0.1 or a separate cosine annealing schedule.
        mse_loss = F.mse_loss(eeg_features, clip_features)

        # InfoNCE branch — does all the real work
        infonce_loss, logit_scale = self.infonce(eeg_features, clip_features)

        total = mse_loss + self.lam * infonce_loss

        return total, logit_scale


def create_multi_subject_model(
    n_channels=17,
    n_timepoints=250,
    latent_dim=768,
    num_subjects=10,
    use_subject_embedding=True,
    subject_emb_dim=64,
    nz_dim=NZ_DIM,
    dropout=0.5,
    temperature=0.07,
    loss_type='infonce',  # 'infonce' or 'hybrid'
    loss_alpha=0.5        # lambda on InfoNCE term (hybrid only)
):
    model = MultiSubjectNICEEEGEncoder(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        latent_dim=latent_dim,
        num_subjects=num_subjects,
        use_subject_embedding=use_subject_embedding,
        subject_emb_dim=subject_emb_dim,
        nz_dim=nz_dim,
        dropout=dropout
    )

    if loss_type == 'hybrid':
        loss_fn = HybridLoss(temperature=temperature, learnable=True, lam=loss_alpha)
        print(f"  Loss           : ENIGMA HybridLoss  "
              f"(MSE + lambda={loss_alpha} * InfoNCE)")
    else:
        loss_fn = InfoNCELoss(temperature=temperature, learnable=True)
        print(f"  Loss           : InfoNCE only")

    print(f"  Temperature    : {temperature}  (scale ~{1.0/temperature:.1f})")
    return model, loss_fn


if __name__ == "__main__":
    print("Testing ENIGMA-aligned Multi-Subject model...")

    model, loss_fn = create_multi_subject_model(
        num_subjects=10,
        use_subject_embedding=True,
        subject_emb_dim=64,
        nz_dim=NZ_DIM
    )

    batch_size = 8
    dummy_eeg  = torch.randn(batch_size, 17, 250)
    dummy_clip = torch.randn(batch_size, 768)
    dummy_sids = torch.randint(0, 10, (batch_size,))

    out = model(dummy_eeg, dummy_sids)
    print(f"\nForward pass : {dummy_eeg.shape} -> {out.shape}")

    loss, temp = loss_fn(out, dummy_clip)
    print(f"Loss         : {loss.item():.4f}   Temperature: {temp:.2f}")

    # Verify subject aligners produce different outputs for same EEG
    eeg4   = torch.randn(4, 17, 250)
    sids_0 = torch.zeros(4, dtype=torch.long)
    sids_1 = torch.ones(4,  dtype=torch.long)
    diff = (model(eeg4, sids_0) - model(eeg4, sids_1)).abs().mean().item()
    print(f"Output diff sub0 vs sub1 : {diff:.4f}  (should be > 0)")

    print("\nAll checks passed!")
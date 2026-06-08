"""
Stage 2 Projector: CLIP/EEG embedding -> LLM token space

Two architectures selectable via projector_type:

  'linear' (Thought2Text baseline):
      (B, 768) -> Linear(768, llm_embed_dim) -> (B, llm_embed_dim)

  'mlp' (improved):
      (B, 768) -> Linear(768, hidden) -> GELU -> Dropout -> Linear(hidden, llm_embed_dim)
                + skip: Linear(768, llm_embed_dim)
                -> LayerNorm(llm_embed_dim)
      The skip connection + LayerNorm mirrors the Stage 1 MLP projector design,
      which proved stable on the same EEG embedding space.

LLM embed dimensions:
    Phi-3.5-mini-instruct  : 3072
    Qwen2.5-1.5B-Instruct  : 1536
    LLaMA-3-8B-Instruct    : 4096  (future)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPtoLLMProjector(nn.Module):
    """
    Projector from CLIP/EEG embedding space to LLM token embedding space.

    projector_type='linear' : single Linear (Thought2Text baseline, 2.36M params for Phi-3.5)
    projector_type='mlp'    : 2-layer MLP with skip + LayerNorm (~8.7M params for Phi-3.5)
    """

    def __init__(self, clip_dim: int = 768, llm_embed_dim: int = 3072,
                 projector_type: str = 'linear', hidden_dim: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.clip_dim = clip_dim
        self.llm_embed_dim = llm_embed_dim
        self.projector_type = projector_type

        if projector_type == 'linear':
            self.linear = nn.Linear(clip_dim, llm_embed_dim)
        elif projector_type == 'mlp':
            self.fc1   = nn.Linear(clip_dim, hidden_dim)
            self.drop  = nn.Dropout(dropout)
            self.fc2   = nn.Linear(hidden_dim, llm_embed_dim)
            self.skip  = nn.Linear(clip_dim, llm_embed_dim)
            self.norm  = nn.LayerNorm(llm_embed_dim)
        else:
            raise ValueError(f"projector_type must be 'linear' or 'mlp', got {projector_type!r}")

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"\nCLIPtoLLMProjector ({projector_type}):")
        if projector_type == 'linear':
            print(f"  Linear({clip_dim}, {llm_embed_dim})")
        else:
            print(f"  Linear({clip_dim}, {hidden_dim}) -> GELU -> Dropout({dropout})")
            print(f"  -> Linear({hidden_dim}, {llm_embed_dim}) + skip Linear({clip_dim}, {llm_embed_dim})")
            print(f"  -> LayerNorm({llm_embed_dim})")
        print(f"  Params: {n_params:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, clip_dim) — L2-normalized CLIP or EEG embedding
        Returns:
            projected: (B, llm_embed_dim)
        """
        if self.projector_type == 'linear':
            return self.linear(x)
        else:
            h = F.gelu(self.fc1(x))
            h = self.drop(h)
            h = self.fc2(h) + self.skip(x)
            return self.norm(h)


def save_projector(projector: CLIPtoLLMProjector, path: str, metadata: dict = None):
    checkpoint = {
        'projector_state_dict': projector.state_dict(),
        'clip_dim': projector.clip_dim,
        'llm_embed_dim': projector.llm_embed_dim,
        'projector_type': projector.projector_type,
        'metadata': metadata or {},
    }
    torch.save(checkpoint, path)
    print(f"  Projector saved to {path}")


def load_projector(path: str, device: str = 'cpu') -> tuple:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    projector_type = checkpoint.get('projector_type', 'linear')
    projector = CLIPtoLLMProjector(
        clip_dim=checkpoint['clip_dim'],
        llm_embed_dim=checkpoint['llm_embed_dim'],
        projector_type=projector_type,
    )
    projector.load_state_dict(checkpoint['projector_state_dict'])
    projector = projector.to(device)
    print(f"  Projector loaded from {path}  (type={projector_type})")
    print(f"  Dims: {checkpoint['clip_dim']} -> {checkpoint['llm_embed_dim']}")
    return projector, checkpoint.get('metadata', {})


if __name__ == "__main__":
    print("Testing CLIPtoLLMProjector...")

    for ptype in ['linear', 'mlp']:
        for llm_dim, name in [(3072, "Phi-3.5-mini"), (1536, "Qwen2.5-1.5B")]:
            proj = CLIPtoLLMProjector(clip_dim=768, llm_embed_dim=llm_dim,
                                      projector_type=ptype)
            dummy = torch.randn(4, 768)
            out = proj(dummy)
            assert out.shape == (4, llm_dim), f"Shape mismatch: {out.shape}"
            print(f"  [{ptype}] {name}: (4,768) -> {out.shape} OK")

    # Save/load round-trip
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        tmp_path = f.name
    proj = CLIPtoLLMProjector(768, 3072, projector_type='mlp')
    save_projector(proj, tmp_path, metadata={"test": True})
    loaded, meta = load_projector(tmp_path)
    diff = (proj.fc1.weight - loaded.fc1.weight).abs().max().item()
    assert diff < 1e-6, f"Weight mismatch after load: {diff}"
    os.unlink(tmp_path)
    print("\nSave/load round-trip passed.")

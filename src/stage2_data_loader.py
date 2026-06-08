"""
Stage 2 Data Loader: CLIP embeddings + captions for projector training

Pairs each CLIP image embedding with its GPT/VLM-generated caption,
tokenizes the caption for the target LLM, and constructs labels with
-100 masking so the cross-entropy loss is computed ONLY on caption tokens
(not on the projected CLIP token position or system/user prompt tokens).

Input format:
    clip_embeddings_image_level.npy  -- (16740, 768), same as Stage 1
    things_captions.json             -- {"0": "caption", ..., "16739": "..."}

Indices:
    0    - 16539 : training images (1654 concepts x 10)
    16540 - 16739: test images     (200 concepts x 1)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CLIPCaptionDataset(Dataset):
    """
    Dataset pairing CLIP embeddings with tokenized captions.

    Each sample returns:
        clip_emb      : (768,) float32, L2-normalized
        input_ids     : (max_length,) long  — full chat template tokens
        attention_mask: (max_length,) long
        labels        : (max_length,) long  — -100 everywhere except caption
        image_idx     : int, original image index in [0, 16739]
    """

    def __init__(
        self,
        clip_embeddings: np.ndarray,
        captions: dict,
        indices: list,
        tokenizer,
        max_length: int = 128,
        system_prompt: str = "You are a helpful vision assistant.",
        user_prompt: str = "Describe this image in one sentence.",
    ):
        self.clip_embeddings = clip_embeddings
        self.captions = captions
        self.indices = indices
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        # Verify all indices have captions
        missing = [i for i in indices if str(i) not in captions]
        if missing:
            raise ValueError(
                f"{len(missing)} indices have no captions. "
                f"First few: {missing[:5]}"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_idx = self.indices[idx]

        # CLIP embedding — L2 normalize (matching multi_subject_data_loader.py line 54)
        clip_emb = self.clip_embeddings[image_idx].astype(np.float32)
        norm = np.linalg.norm(clip_emb) + 1e-8
        clip_emb = clip_emb / norm
        clip_emb = torch.tensor(clip_emb, dtype=torch.float32)

        caption = self.captions[str(image_idx)]

        # Build full conversation (system + user + assistant/caption)
        messages_full = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",    "content": self.user_prompt},
            {"role": "assistant", "content": caption},
        ]
        # Build prompt-only (no assistant response) to find caption boundary
        messages_prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",    "content": self.user_prompt},
        ]

        full_text = self.tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize full sequence
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = full_enc['input_ids'].squeeze(0)         # (max_length,)
        attention_mask = full_enc['attention_mask'].squeeze(0)

        # Find where caption tokens start (prompt boundary)
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=False,
        )
        prompt_len = prompt_enc['input_ids'].shape[1]

        # Labels: -100 for prompt tokens and padding; loss only on caption
        labels = input_ids.clone()
        labels[:prompt_len] = -100                           # mask prompt
        labels[attention_mask == 0] = -100                   # mask padding

        # The projected CLIP embedding will be prepended at position 0
        # in the training loop, so labels will gain a leading -100 there.

        return {
            'clip_emb':       clip_emb,
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
            'image_idx':      image_idx,
        }


def create_stage2_dataloaders(
    clip_embeddings_path: str,
    captions_path: str,
    tokenizer,
    batch_size: int = 16,
    val_split: float = 0.1,
    max_length: int = 128,
    num_workers: int = 0,
    system_prompt: str = "You are a helpful vision assistant.",
    user_prompt: str = "Describe this image in one sentence.",
    seed: int = 42,
):
    """
    Create train, val, and test dataloaders for Stage 2.

    Returns:
        train_loader, val_loader, test_loader
    """
    print("=" * 70)
    print("CREATING STAGE 2 DATALOADERS")
    print("=" * 70)

    # Load CLIP embeddings
    clip_embeddings = np.load(clip_embeddings_path)
    print(f"CLIP embeddings: {clip_embeddings.shape}")

    # Load captions
    with open(captions_path, 'r') as f:
        captions = json.load(f)
    print(f"Captions loaded: {len(captions)} entries")

    # Training indices: 0-16539; test indices: 16540-16739
    train_val_indices = list(range(16540))
    test_indices = list(range(16540, 16740))

    # Shuffle and split train/val
    rng = np.random.default_rng(seed)
    rng.shuffle(train_val_indices)
    n_val = int(len(train_val_indices) * val_split)
    val_indices = train_val_indices[:n_val]
    train_indices = train_val_indices[n_val:]

    print(f"\nSplit:")
    print(f"  Train : {len(train_indices)}")
    print(f"  Val   : {len(val_indices)}")
    print(f"  Test  : {len(test_indices)}")

    dataset_kwargs = dict(
        clip_embeddings=clip_embeddings,
        captions=captions,
        tokenizer=tokenizer,
        max_length=max_length,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    train_dataset = CLIPCaptionDataset(indices=train_indices, **dataset_kwargs)
    val_dataset   = CLIPCaptionDataset(indices=val_indices,   **dataset_kwargs)
    test_dataset  = CLIPCaptionDataset(indices=test_indices,  **dataset_kwargs)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    print(f"\nDataloaders created.")
    print("=" * 70)
    return train_loader, val_loader, test_loader

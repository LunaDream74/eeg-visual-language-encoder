"""
Stage 2 Evaluation: Generate captions and score against ground truth

Modes:
  1. CLIP mode (default): CLIP embeddings -> projector -> LLM -> captions
  2. EEG mode (--use_eeg): EEG signals -> Stage 1 encoder -> projector -> LLM -> captions
     This is the full brain-to-text pipeline (no images at inference time).

Metrics (Thought2Text paper):
  BLEU-1    : ~25% target
  ROUGE-1 F1: ~30% target
  BERTScore F1: ~0.89 target

Usage (CLIP mode):
  python eval_stage2.py \\
    --projector_path checkpoints_stage2/best_projector.pth \\
    --llm_name microsoft/Phi-3.5-mini-instruct \\
    --captions_path things_captions.json

Usage (full EEG pipeline):
  python eval_stage2.py \\
    --projector_path checkpoints_stage2/best_projector.pth \\
    --llm_name microsoft/Phi-3.5-mini-instruct \\
    --captions_path things_captions.json \\
    --use_eeg \\
    --eeg_checkpoint checkpoints_multi/best_multi_subject_model.pth
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from stage2_projector import load_projector


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

def _build_initial_embeds(projector, llm, tokenizer, clip_embedding, device,
                           system_prompt, user_prompt):
    """Shared setup for greedy and beam search: returns (inputs_embeds, attention_mask)."""
    projector.eval()

    if clip_embedding.dim() == 1:
        clip_embedding = clip_embedding.unsqueeze(0)
    clip_embedding = clip_embedding.to(device)

    projected = projector(clip_embedding)              # (1, D)
    projected_token = projected.unsqueeze(1)           # (1, 1, D)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_enc = tokenizer(
        prompt_text, return_tensors='pt', add_special_tokens=False,
    ).to(device)

    token_embeds = llm.get_input_embeddings()(prompt_enc['input_ids'])  # (1, L, D)
    projected_token = projected_token.to(token_embeds.dtype)
    inputs_embeds = torch.cat([projected_token, token_embeds], dim=1)   # (1, 1+L, D)

    clip_mask = torch.ones(1, 1, dtype=torch.long, device=device)
    attention_mask = torch.cat([clip_mask, prompt_enc['attention_mask']], dim=1)
    return inputs_embeds, attention_mask


@torch.no_grad()
def generate_caption(
    projector,
    llm,
    tokenizer,
    clip_embedding: torch.Tensor,
    device,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 50,
    num_beams: int = 1,
) -> str:
    """
    Generate a caption from a single CLIP/EEG embedding.

    num_beams=1 : greedy decode (fast, ~5s/image)
    num_beams>1 : manual beam search without KV cache (~num_beams x slower but better)

    Both paths avoid generate() + inputs_embeds KV cache bugs in Phi-3.5
    (position_ids shape mismatch on step 2+).
    """
    inputs_embeds, attention_mask = _build_initial_embeds(
        projector, llm, tokenizer, clip_embedding, device, system_prompt, user_prompt
    )

    if num_beams <= 1:
        return _greedy_decode(llm, tokenizer, inputs_embeds, attention_mask,
                              device, max_new_tokens)
    else:
        return _beam_search(llm, tokenizer, inputs_embeds, attention_mask,
                            device, max_new_tokens, num_beams)


def _greedy_decode(llm, tokenizer, inputs_embeds, attention_mask,
                   device, max_new_tokens):
    generated_ids = []
    current_embeds = inputs_embeds
    current_mask = attention_mask

    for _ in range(max_new_tokens):
        out = llm(inputs_embeds=current_embeds, attention_mask=current_mask,
                  use_cache=False)
        next_token_id = out.logits[:, -1, :].argmax(dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token_id.item())

        next_embed = llm.get_input_embeddings()(next_token_id.unsqueeze(0))
        next_embed = next_embed.to(current_embeds.dtype)
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)
        current_mask = torch.cat(
            [current_mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1
        )

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _beam_search(llm, tokenizer, inputs_embeds, attention_mask,
                 device, max_new_tokens, num_beams):
    """
    Manual beam search without KV cache.
    Each beam: (cumulative_log_prob, token_ids, current_embeds, current_mask)
    Score = cumulative_log_prob / length  (length-normalised to avoid short-caption bias)
    """
    beams = [(0.0, [], inputs_embeds, attention_mask)]
    completed = []

    for _ in range(max_new_tokens):
        if not beams:
            break
        candidates = []
        for log_prob, tokens, embeds, mask in beams:
            out = llm(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
            logits = out.logits[:, -1, :]                    # (1, vocab)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            top_lp, top_ids = log_probs.topk(num_beams, dim=-1)  # (1, B)

            for i in range(num_beams):
                tid = top_ids[0, i].item()
                new_lp = log_prob + top_lp[0, i].item()
                if tid == tokenizer.eos_token_id:
                    if tokens:
                        completed.append((new_lp / len(tokens), tokens))
                else:
                    new_embed = llm.get_input_embeddings()(
                        torch.tensor([[tid]], device=device)
                    ).to(embeds.dtype)
                    new_embeds = torch.cat([embeds, new_embed], dim=1)
                    new_mask = torch.cat(
                        [mask, torch.ones(1, 1, dtype=torch.long, device=device)], dim=1
                    )
                    candidates.append((new_lp, tokens + [tid], new_embeds, new_mask))

        # Keep top num_beams by length-normalised score
        candidates.sort(
            key=lambda x: x[0] / max(len(x[1]), 1), reverse=True
        )
        beams = candidates[:num_beams]

    # Drain remaining beams into completed
    for log_prob, tokens, _, _ in beams:
        if tokens:
            completed.append((log_prob / len(tokens), tokens))

    if not completed:
        return ""

    completed.sort(key=lambda x: x[0], reverse=True)
    return tokenizer.decode(completed[0][1], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_bleu1(hypotheses: list, references: list) -> float:
    """BLEU-1 (unigram)."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    refs = [[ref.lower().split()] for ref in references]
    hyps = [hyp.lower().split() for hyp in hypotheses]
    smoothie = SmoothingFunction().method1
    return corpus_bleu(refs, hyps, weights=(1, 0, 0, 0),
                       smoothing_function=smoothie) * 100


def compute_rouge1(hypotheses: list, references: list) -> float:
    """ROUGE-1 F1 (average)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = [
        scorer.score(ref, hyp)['rouge1'].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    return np.mean(scores) * 100


def compute_bertscore(hypotheses: list, references: list, device: str) -> float:
    """BERTScore F1 (average)."""
    from bert_score import score as bs_score
    _, _, F = bs_score(
        hypotheses, references,
        lang='en', device=device, verbose=False
    )
    return F.mean().item()


# ---------------------------------------------------------------------------
# EEG pipeline helpers
# ---------------------------------------------------------------------------

def load_eeg_encoder(checkpoint_path: str, device):
    """Load frozen Stage 1 EEG encoder."""
    from multi_subject_architecture import create_multi_subject_model

    ckpt = torch.load(checkpoint_path, map_location='cpu')
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
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    print(f"EEG encoder loaded from {checkpoint_path}")
    print(f"  Subjects: {subjects}  Test Top-1: {ckpt.get('test_top1', '?')}%")
    return model, subjects


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main(args):
    print("\n" + "=" * 70)
    print("STAGE 2 EVALUATION")
    print("=" * 70)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Device: {device}")

    # Load captions (ground truth)
    with open(args.captions_path, 'r') as f:
        gt_captions = json.load(f)
    print(f"Ground truth captions: {len(gt_captions)}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.llm_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LLM (4-bit, frozen)
    print(f"Loading LLM (4-bit): {args.llm_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False

    # Load projector
    print(f"\nLoading projector: {args.projector_path}")
    projector, proj_meta = load_projector(args.projector_path, device=device_str)
    projector.eval()
    print(f"  Trained with: {proj_meta.get('llm_name', 'unknown')}")
    print(f"  Best val loss: {proj_meta.get('val_loss', '?')}")

    # ------------------------------------------------------------------
    # Get CLIP embeddings for evaluation
    # ------------------------------------------------------------------
    if args.use_eeg:
        # Full EEG pipeline: EEG -> Stage 1 encoder -> CLIP-aligned emb -> projector -> text
        print("\n--- EEG MODE: Full brain-to-text pipeline ---")
        assert args.eeg_checkpoint, "--eeg_checkpoint required for --use_eeg"

        eeg_encoder, subjects = load_eeg_encoder(args.eeg_checkpoint, device)

        from multi_subject_data_loader import create_multi_subject_dataloaders
        _, _, test_loader, _ = create_multi_subject_dataloaders(
            preprocessed_path=args.preprocessed_path,
            clip_embeddings_path=args.clip_embeddings_path,
            subjects=subjects,
            batch_size=args.batch_size,
            num_workers=0,
        )

        # Collect EEG-derived embeddings for test images
        all_embs = []
        all_img_ids = []
        with torch.no_grad():
            for batch in test_loader:
                eeg = batch['eeg'].to(device)
                sids = batch['subject_id'].to(device)
                emb = eeg_encoder(eeg, sids)
                emb = F.normalize(emb, dim=1)
                all_embs.append(emb.cpu())
                all_img_ids.extend(batch['image_id'].tolist())
        all_embs = torch.cat(all_embs, dim=0)

        # Average embeddings across all subjects per image (test-time ensembling).
        # Reduces noise vs. taking a single subject's EEG embedding.
        # --no_avg_subjects falls back to first-occurrence (single subject).
        from collections import defaultdict
        accum = defaultdict(list)
        for emb, img_id in zip(all_embs, all_img_ids):
            accum[img_id].append(emb)

        eval_image_ids = sorted(accum.keys())
        if args.avg_subjects:
            # Mean across all subjects, then re-normalize to unit sphere
            stacked = [torch.stack(accum[i]).mean(0) for i in eval_image_ids]
            eval_embeddings = torch.stack(stacked)
            eval_embeddings = F.normalize(eval_embeddings, dim=1)
            n_per_img = len(accum[eval_image_ids[0]])
            print(f"Unique test images: {len(eval_image_ids)}  "
                  f"(averaged {n_per_img} subjects per image)")
        else:
            # First subject only (original behaviour)
            eval_embeddings = torch.stack([accum[i][0] for i in eval_image_ids])
            print(f"Unique test images: {len(eval_image_ids)}  (single subject, no averaging)")

    else:
        # Standard CLIP mode: load precomputed CLIP embeddings
        print("\n--- CLIP MODE ---")
        clip_embeddings = np.load(args.clip_embeddings_path)

        # Use test images (16540-16739) or a random subset of training images
        if args.eval_split == 'test':
            eval_image_ids = list(range(16540, 16740))
        else:
            rng = np.random.default_rng(42)
            eval_image_ids = rng.choice(16540, size=args.n_eval_samples,
                                        replace=False).tolist()
        eval_image_ids = sorted(eval_image_ids)

        embs = clip_embeddings[eval_image_ids].astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        eval_embeddings = torch.tensor(embs / norms)
        print(f"Eval images: {len(eval_image_ids)}")

    # ------------------------------------------------------------------
    # Generate captions
    # ------------------------------------------------------------------
    decode_mode = f"beam_search(width={args.num_beams})" if args.num_beams > 1 else "greedy"
    print(f"\nGenerating captions (max_new_tokens={args.max_new_tokens}, decode={decode_mode})...")
    generated_captions = []
    reference_captions = []

    for i, img_id in enumerate(eval_image_ids):
        clip_emb = eval_embeddings[i]
        gen = generate_caption(
            projector, llm, tokenizer, clip_emb, device,
            system_prompt=args.system_prompt,
            user_prompt=args.user_prompt,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        generated_captions.append(gen)
        reference_captions.append(gt_captions.get(str(img_id), ""))

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(eval_image_ids)}]  {gen[:80]}")

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("METRICS")
    print("=" * 70)

    bleu1 = compute_bleu1(generated_captions, reference_captions)
    rouge1 = compute_rouge1(generated_captions, reference_captions)
    print(f"BLEU-1  : {bleu1:.2f}%   (Thought2Text target: ~25%)")
    print(f"ROUGE-1 : {rouge1:.2f}%   (Thought2Text target: ~30%)")

    if args.compute_bertscore:
        bscore = compute_bertscore(generated_captions, reference_captions, device_str)
        print(f"BERTScore F1: {bscore:.4f}   (Thought2Text target: ~0.89)")
    else:
        print("BERTScore: skipped (--compute_bertscore to enable)")

    # ------------------------------------------------------------------
    # Sample outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUTS")
    print("=" * 70)
    import random
    sample_indices = random.sample(range(len(eval_image_ids)), min(10, len(eval_image_ids)))
    for si in sorted(sample_indices):
        img_id = eval_image_ids[si]
        print(f"\n  Image {img_id}:")
        print(f"    GT  : {reference_captions[si]}")
        print(f"    GEN : {generated_captions[si]}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.output_file:
        results = {
            "mode": "eeg" if args.use_eeg else "clip",
            "llm_name": args.llm_name,
            "projector_path": args.projector_path,
            "n_evaluated": len(eval_image_ids),
            "bleu1": bleu1,
            "rouge1": rouge1,
            "samples": [
                {"image_id": eval_image_ids[i],
                 "reference": reference_captions[i],
                 "generated": generated_captions[i]}
                for i in range(len(eval_image_ids))
            ]
        }
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 evaluation")

    # Required
    parser.add_argument('--projector_path', type=str, required=True,
                        help='Path to best_projector.pth')
    parser.add_argument('--llm_name', type=str,
                        default='microsoft/Phi-3.5-mini-instruct')
    parser.add_argument('--captions_path', type=str,
                        default='things_captions.json',
                        help='Ground truth captions JSON')

    # Data
    parser.add_argument('--clip_embeddings_path', type=str,
                        default='THINGS_clip_embeddings/clip_embeddings_image_level.npy')
    parser.add_argument('--eval_split', type=str, default='test',
                        choices=['test', 'train'],
                        help='Which split to evaluate on')
    parser.add_argument('--n_eval_samples', type=int, default=200,
                        help='Number of training samples to eval (if eval_split=train)')
    parser.add_argument('--batch_size', type=int, default=32)

    # Generation
    parser.add_argument('--system_prompt', type=str,
                        default='You are a helpful vision assistant.')
    parser.add_argument('--user_prompt', type=str,
                        default='Describe this image in one sentence.')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--num_beams', type=int, default=1,
                        help='Beam width for generation. 1=greedy (fast), '
                             '4-5=beam search (better quality, ~num_beams x slower)')

    # Metrics
    parser.add_argument('--compute_bertscore', action='store_true',
                        help='Compute BERTScore (slower, requires bert-score package)')

    # EEG mode
    parser.add_argument('--use_eeg', action='store_true',
                        help='Use Stage 1 EEG encoder instead of CLIP embeddings')
    parser.add_argument('--avg_subjects', action='store_true', default=True,
                        help='Average EEG embeddings across all subjects per image '
                             '(test-time ensembling, default: True). '
                             'Use --no_avg_subjects for single-subject mode.')
    parser.add_argument('--no_avg_subjects', dest='avg_subjects', action='store_false',
                        help='Use only first subject EEG embedding (no averaging)')
    parser.add_argument('--eeg_checkpoint', type=str,
                        default='checkpoints_multi/best_multi_subject_model.pth')
    parser.add_argument('--preprocessed_path', type=str,
                        default='./preprocessed_data_250Hz')

    # Output
    parser.add_argument('--output_file', type=str, default=None,
                        help='Save results JSON (optional)')

    args = parser.parse_args()
    main(args)

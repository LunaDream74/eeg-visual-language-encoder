"""
Multi-Subject NICE-EEG Training Script

Train EEG encoder on multiple subjects simultaneously
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from collections import defaultdict

from multi_subject_data_loader import create_multi_subject_dataloaders
from multi_subject_architecture import create_multi_subject_model
from training_logger import log_best_model, log_best_model_json, print_training_summary


def evaluate_loss(model, dataloader, loss_fn, device):
    """Evaluate validation loss (like NICE paper)"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            clip_target = batch['clip_emb'].to(device)
            subject_ids = batch['subject_id'].to(device)
            
            pred_emb = model(eeg, subject_ids)
            loss, _ = loss_fn(pred_emb, clip_target)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, clip_embeddings_all, device, is_test=False):
    """Evaluate with zero-shot retrieval (for monitoring only)"""
    model.eval()
    
    all_pred = []
    all_ids = []
    all_subject_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            subject_ids = batch['subject_id'].to(device)
            image_ids = batch['image_id']
            
            pred_emb = model(eeg, subject_ids)
            pred_emb = F.normalize(pred_emb, dim=1)
            
            all_pred.append(pred_emb.cpu())
            all_ids.append(image_ids)
            all_subject_ids.append(batch['subject_id'])
    
    all_pred = torch.cat(all_pred, dim=0)
    all_ids = torch.cat(all_ids, dim=0)
    all_subject_ids = torch.cat(all_subject_ids, dim=0)
    
    if is_test:
        # Search against test images (16540-16739)
        clip_test = clip_embeddings_all[16540:16740]
        clip_test = torch.tensor(clip_test, dtype=torch.float32)
        clip_test = F.normalize(clip_test, dim=1)
        similarities = torch.mm(all_pred, clip_test.t())
        top1_preds = similarities.argmax(dim=1) + 16540
        top5_preds = similarities.topk(5, dim=1)[1] + 16540
    else:
        # Search against train images (0-16539)
        clip_train = clip_embeddings_all[:16540]
        clip_train = torch.tensor(clip_train, dtype=torch.float32)
        clip_train = F.normalize(clip_train, dim=1)
        similarities = torch.mm(all_pred, clip_train.t())
        top1_preds = similarities.argmax(dim=1)
        top5_preds = similarities.topk(5, dim=1)[1]
    
    # Overall accuracy
    top1_acc = (top1_preds == all_ids).float().mean().item() * 100
    top5_acc = torch.any(top5_preds == all_ids.unsqueeze(1), dim=1).float().mean().item() * 100
    
    # Per-subject accuracy
    per_subject_acc = {}
    for subject_id in all_subject_ids.unique():
        mask = (all_subject_ids == subject_id)
        subj_top1 = (top1_preds[mask] == all_ids[mask]).float().mean().item() * 100
        subj_top5 = torch.any(top5_preds[mask] == all_ids[mask].unsqueeze(1), dim=1).float().mean().item() * 100
        per_subject_acc[subject_id.item()] = (subj_top1, subj_top5)
    
    return top1_acc, top5_acc, per_subject_acc


def train_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        eeg = batch['eeg'].to(device)
        clip_target = batch['clip_emb'].to(device)
        subject_ids = batch['subject_id'].to(device)
        
        pred_emb = model(eeg, subject_ids)
        loss, temp = loss_fn(pred_emb, clip_target)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}, Temp: {temp:.2f}")
    
    return total_loss / len(train_loader)


def main(args):
    print("\n" + "="*70)
    print("MULTI-SUBJECT NICE-EEG TRAINING")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Parse subjects
    if args.subjects == 'all':
        subjects = list(range(1, 11))
    else:
        subjects = [int(s) for s in args.subjects.split(',')]
    
    print(f"Training on subjects: {subjects}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, num_subjects = create_multi_subject_dataloaders(
        preprocessed_path=args.preprocessed_path,
        clip_embeddings_path=args.clip_embeddings_path,
        subjects=subjects,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load CLIP embeddings
    # Training embeddings (may be blurred) — used as targets during training
    clip_embeddings_all = np.load(args.clip_embeddings_path)
    clip_dim = clip_embeddings_all.shape[1]

    # Evaluation embeddings — always sharp, used for retrieval at test time
    if args.eval_clip_embeddings_path:
        clip_embeddings_eval = np.load(args.eval_clip_embeddings_path)
        print(f"Train embeddings : {args.clip_embeddings_path}  {clip_embeddings_all.shape}")
        print(f"Eval embeddings  : {args.eval_clip_embeddings_path}  {clip_embeddings_eval.shape}")
        assert clip_embeddings_eval.shape[0] == clip_embeddings_all.shape[0], \
            "Train and eval embeddings must have same number of images"
    else:
        clip_embeddings_eval = clip_embeddings_all   # same file for both
    
    # Create model
    model, loss_fn = create_multi_subject_model(
        n_channels=17,
        n_timepoints=250,
        latent_dim=clip_dim,
        num_subjects=len(subjects),
        use_subject_embedding=args.use_subject_embedding,
        subject_emb_dim=args.subject_emb_dim,
        dropout=0.5,
        nz_dim=args.nz_dim,
        loss_type=args.loss_type,
        loss_alpha=args.loss_alpha
    )
    model = model.to(device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01  # Add regularization
    )
    
    # Learning rate warmup + cosine schedule
    def get_lr_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
        """LR warmup then cosine decay"""
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return epoch / warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    warmup_epochs = min(10, args.epochs // 10)  # 10 epochs or 10% of total
    scheduler = get_lr_schedule_with_warmup(optimizer, warmup_epochs, args.epochs)
    
    print(f"\nOptimization:")
    print(f"  Optimizer: AdamW (weight_decay=0.01)")
    print(f"  Initial LR: {args.lr:.2e}")
    print(f"  LR warmup: {warmup_epochs} epochs")
    print(f"  LR schedule: Cosine decay")
    
    # Training loop
    best_val_loss = float('inf')  # NICE paper: save on minimum validation loss!
    best_val_top1 = 0
    best_val_top5 = 0
    best_test_top1 = 0
    best_test_top5 = 0
    best_epoch = 0
    patience_counter = 0
    
    print("\n" + "="*70)
    print("TRAINING START (Using Validation Loss for Model Selection)")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} (LR: {optimizer.param_groups[0]['lr']:.2e})")
        
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
        scheduler.step()
        
        print(f"Train loss: {train_loss:.4f}")
        
        # Validation LOSS (for model selection - like NICE paper!)
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        train_val_gap = val_loss - train_loss
        print(f"Validation loss: {val_loss:.4f} (gap: {train_val_gap:+.4f})")
        
        if train_val_gap > 1.0:
            print(f"⚠️  Warning: Large train-val gap ({train_val_gap:.2f}) - possible overfitting!")
        
        # Validation & Test ACCURACY (for monitoring only)
        val_top1, val_top5, val_per_subject = evaluate(model, val_loader, clip_embeddings_eval, device)
        print(f"Validation accuracy: Top-1 {val_top1:.2f}%, Top-5 {val_top5:.2f}% (monitoring only)")
        
        test_top1, test_top5, test_per_subject = evaluate(model, test_loader, clip_embeddings_eval, device, is_test=True)
        print(f"Test accuracy: Top-1 {test_top1:.2f}%, Top-5 {test_top5:.2f}%")
        
        # Per-subject results (show top 3 and bottom 3)
        if len(test_per_subject) > 0:
            sorted_subjects = sorted(test_per_subject.items(), key=lambda x: x[1][0], reverse=True)
            print(f"\nPer-Subject Test Top-1:")
            for subj_id, (top1, top5) in sorted_subjects[:3]:
                print(f"  Subject {subj_id+1}: {top1:.2f}%")
            if len(sorted_subjects) > 6:
                print("  ...")
                for subj_id, (top1, top5) in sorted_subjects[-3:]:
                    print(f"  Subject {subj_id+1}: {top1:.2f}%")
        
        # Save best based on VALIDATION LOSS (NICE paper approach!)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_top1 = val_top1
            best_val_top5 = val_top5
            best_test_top1 = test_top1
            best_test_top5 = test_top5
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_top1': val_top1,
                'val_top5': val_top5,
                'test_top1': test_top1,
                'test_top5': test_top5,
                'test_per_subject': test_per_subject,
                'subjects': subjects
            }
            
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_multi_subject_model.pth'))
            
            print(f"✓ NEW BEST! Val Loss: {val_loss:.4f}, Test Top-1: {test_top1:.2f}%")
            
            # Log
            log_best_model(
                log_file=os.path.join(args.output_dir, 'training_results.txt'),
                subject='multi' if len(subjects) > 1 else subjects[0],
                architecture=args.architecture,
                best_epoch=epoch,
                val_top1=val_top1,
                val_top5=val_top5,
                test_top1=test_top1,
                test_top5=test_top5,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                total_epochs=args.epochs,
                notes=f"{len(subjects)} subjects: {subjects}" + (f" | {args.notes}" if args.notes else "")
            )
            
            log_best_model_json(
                log_file=os.path.join(args.output_dir, 'training_results.json'),
                subject='multi' if len(subjects) > 1 else subjects[0],
                architecture=args.architecture,
                best_epoch=epoch,
                val_top1=val_top1,
                val_top5=val_top5,
                test_top1=test_top1,
                test_top5=test_top5,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                total_epochs=args.epochs,
                notes=f"{len(subjects)} subjects: {subjects}" + (f" | {args.notes}" if args.notes else "")
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (val loss not improving)")
                break
    
    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Test Top-1: {best_test_top1:.2f}%")
    print(f"Best Test Top-5: {best_test_top5:.2f}%")
    print(f"\nNote: Model saved based on VALIDATION LOSS (NICE paper approach)")
    print(f"Validation accuracy (0.44%) looks low but is 73× random chance!")
    print(f"\nNICE paper (average): ~10-12%")
    print(f"Your performance: {best_test_top1:.2f}% ({best_test_top1/10.4*100:.1f}% of NICE baseline)")
    
    # Show history if available
    log_path = os.path.join(args.output_dir, 'training_results.txt')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            num_results = sum(1 for line in f if line.strip())
        
        if num_results > 1:
            print("\n" + "="*70)
            print("TRAINING HISTORY SUMMARY")
            print("="*70)
            print_training_summary(log_path, top_n=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_path', type=str, default='./preprocessed_data_250Hz')
    parser.add_argument('--clip_embeddings_path', type=str,
                       default='THINGS_clip_embeddings/clip_embeddings_image_level.npy',
                       help='CLIP embeddings used as training targets. '
                            'Use clip_embeddings_blur_768.npy for UBP blur training.')
    parser.add_argument('--eval_clip_embeddings_path', type=str, default=None,
                       help='CLIP embeddings for evaluation/retrieval (always sharp). '
                            'If not set, uses --clip_embeddings_path for both.')
    parser.add_argument('--nz_dim', type=int, default=184,
                       help='Bottleneck dim (Nz). Default 184 matches ENIGMA.')

    # ── Subject selection ────────────────────────────────────────────────────
    parser.add_argument('--subjects', type=str, default='all',
                       help='Comma-separated subject IDs (e.g. "1,2,3") or "all".')
    parser.add_argument('--subject', type=int, default=None,
                       help='Shorthand for single-subject run. "--subject 3" == "--subjects 3"')

    parser.add_argument('--use_subject_embedding', action='store_true', default=True)
    parser.add_argument('--subject_emb_dim', type=int, default=64)
    parser.add_argument('--architecture', type=str, default='MultiSubject-NICE-EEG')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./checkpoints_multi_incase_sthbroke')
    parser.add_argument('--loss_type', type=str, default='infonce',
                       choices=['infonce', 'hybrid'],
                       help='infonce: best performing. hybrid: ENIGMA MSE+InfoNCE.')
    parser.add_argument('--loss_alpha', type=float, default=0.5,
                       help='Lambda for hybrid loss. Default 0.5 matches ENIGMA.')
    parser.add_argument('--notes', type=str, default=None)

    args = parser.parse_args()

    # ── --subject shorthand ──────────────────────────────────────────────────
    if args.subject is not None:
        args.subjects = str(args.subject)

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
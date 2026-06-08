"""
Training Results Logger

Logs best model performance to a text file for easy tracking across multiple runs.
"""

import os
from datetime import datetime
import json

def log_best_model(
    log_file='training_results.txt',
    subject=None,
    architecture='NICE-EEG',
    best_epoch=None,
    val_top1=None,
    val_top5=None,
    test_top1=None,
    test_top5=None,
    batch_size=None,
    learning_rate=None,
    total_epochs=None,
    train_loss=None,
    val_loss=None,
    notes=None
):
    """
    Log best model performance to a text file
    
    Args:
        log_file: Path to log file
        subject: Subject ID (e.g., 2) or 'multi' for multi-subject
        architecture: Model architecture name
        best_epoch: Epoch where best model was found
        val_top1: Validation Top-1 accuracy (%)
        val_top5: Validation Top-5 accuracy (%)
        test_top1: Test Top-1 accuracy (%)
        test_top5: Test Top-5 accuracy (%)
        batch_size: Batch size used
        learning_rate: Learning rate used
        total_epochs: Total epochs trained
        train_loss: Training loss at best epoch
        val_loss: Validation loss at best epoch
        notes: Optional notes
    """
    
    
    timestamp = datetime.now().isoformat()
    
    # Create log entry
    log_entry = {
        'timestamp': timestamp,
        'subject': subject,
        'architecture': architecture,
        'best_epoch': best_epoch,
        'total_epochs': total_epochs,
        'val_top1': val_top1,
        'val_top5': val_top5,
        'test_top1': test_top1,
        'test_top5': test_top5,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'notes': notes
    }
    
    # Format as single line
    train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
    val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
    
    log_line = (
        f"subject: {subject} "
        f"architecture: {architecture} "
        f"best_epoch: {best_epoch} "
        f"train_loss: {train_loss_str} "
        f"val_loss: {val_loss_str} "
        f"val_top1: {val_top1:.4f} "
        f"val_top5: {val_top5:.4f} "
        f"test_top1: {test_top1:.4f} "
        f"test_top5: {test_top5:.4f} "
        f"batch_size: {batch_size} "
        f"lr: {learning_rate} "
        f"total_epochs: {total_epochs} "
        f"timestamp: {timestamp}"
    )
    
    if notes:
        log_line += f" notes: {notes}"
    
    # Append to log file
    with open(log_file, 'a') as f:
        f.write(log_line + '\n')
    
    print(f"\n✓ Results logged to: {log_file}")
    
    return log_entry


def log_best_model_json(
    log_file='training_results.json',
    **kwargs
):
    """
    Log best model performance to a JSON file (easier to parse)
    
    Args:
        log_file: Path to JSON log file
        **kwargs: Same arguments as log_best_model()
    """
    
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        'timestamp': timestamp,
        **kwargs
    }
    
    # Read existing logs
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                content = f.read().strip()
            logs = json.loads(content) if content else []
        except (json.JSONDecodeError, ValueError):
            # File is corrupted or empty — start fresh, back up the bad file
            backup = log_file + '.corrupted'
            os.rename(log_file, backup)
            print(f"⚠️  Corrupted JSON backed up to: {backup}")
            logs = []
    else:
        logs = []
    
    # Append new entry
    logs.append(log_entry)
    
    # Write back
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"✓ Results logged to: {log_file}")
    
    return log_entry


def print_training_summary(log_file='training_results.json', top_n=10):
    """
    Print a summary of best results from log file
    
    Args:
        log_file: Path to log file
        top_n: Show top N results by test accuracy
    """
    
    if not os.path.exists(log_file):
        print(f"\nNo training history yet. Results will be logged to: {log_file}")
        return
    
    # Parse log file
    results = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                # Parse key-value pairs
                entry = {}
                for pair in line.split():
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        entry[key] = value
                
                # Only add if has test_top1 (skip incomplete entries)
                if 'test_top1' in entry and entry['test_top1']:
                    results.append(entry)
    
    if not results:
        print(f"\nNo results in log file yet. Train to completion to see history!")
        return
    
    # Sort by test_top1 (handle conversion errors)
    def safe_float(x):
        try:
            return float(x.get('test_top1', 0))
        except (ValueError, TypeError):
            return 0.0
    
    results.sort(key=safe_float, reverse=True)
    
    # Print summary
    print("\n" + "="*120)
    print(f"TOP {min(top_n, len(results))} TRAINING RESULTS (by Test Top-1)")
    print("="*120)
    print(f"{'Rank':<5} {'Subject':<8} {'Architecture':<15} {'Epoch':<6} "
          f"{'Train Loss':<11} {'Val Loss':<11} {'Val T1':<8} {'Test T1':<8} {'Test T5':<8} {'Batch':<6} {'LR':<8} {'Date':<12}")
    print("-"*120)
    
    for i, result in enumerate(results[:top_n], 1):
        subject = result.get('subject', 'N/A')
        arch = result.get('architecture', 'N/A')[:14]
        epoch = result.get('best_epoch', 'N/A')
        train_loss = result.get('train_loss', 'N/A')
        val_loss = result.get('val_loss', 'N/A')
        val_t1 = result.get('val_top1', 'N/A')
        test_t1 = result.get('test_top1', 'N/A')
        test_t5 = result.get('test_top5', 'N/A')
        batch = result.get('batch_size', 'N/A')
        lr = result.get('lr', 'N/A')
        timestamp = result.get('timestamp', 'N/A')[:10]
        
        print(f"{i:<5} {subject:<8} {arch:<15} {epoch:<6} "
              f"{train_loss:<11} {val_loss:<11} {val_t1:<8} {test_t1:<8} {test_t5:<8} {batch:<6} {lr:<8} {timestamp:<12}")
    
    print("="*120)


if __name__ == "__main__":
    # Example usage
    log_best_model(
        log_file='training_results.txt',
        subject=2,
        architecture='NICE-EEG',
        best_epoch=84,
        val_top1=0.36,
        val_top5=1.33,
        test_top1=10.50,
        test_top5=31.50,
        batch_size=512,
        learning_rate=3e-4,
        total_epochs=100,
        notes='Fixed temperature to 14.3'
    )
    
    # Also log to JSON
    log_best_model_json(
        log_file='training_results.json',
        subject=2,
        architecture='NICE-EEG',
        best_epoch=84,
        val_top1=0.36,
        val_top5=1.33,
        test_top1=10.50,
        test_top5=31.50,
        batch_size=512,
        learning_rate=3e-4,
        total_epochs=100,
        notes='Fixed temperature to 14.3'
    )
    
    # Print summary
    print_training_summary('training_results.json')
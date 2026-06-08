"""
Analyze Training Results

Reads training_results.txt/json and creates analysis/visualizations
"""

import json
import os
from collections import defaultdict


def analyze_results(log_file='checkpoints/training_results.txt'):
    """Analyze training results from log file"""
    
    if not os.path.exists(log_file):
        print(f"No log file found: {log_file}")
        return
    
    results = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                entry = {}
                for pair in line.split():
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        entry[key] = value
                
                # Only add if has test_top1
                if 'test_top1' in entry and entry['test_top1']:
                    results.append(entry)
    
    if not results:
        print("No results found in log file")
        return
    
    # Helper to safely convert to float
    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    print("\n" + "="*100)
    print("TRAINING RESULTS ANALYSIS")
    print("="*100)
    
    # Overall statistics
    print(f"\nTotal training runs: {len(results)}")
    
    # Best results
    best_test = max(results, key=lambda x: safe_float(x.get('test_top1', 0)))
    print(f"\n🏆 BEST TEST RESULT:")
    print(f"  Subject: {best_test.get('subject', 'N/A')}")
    print(f"  Architecture: {best_test.get('architecture', 'N/A')}")
    print(f"  Epoch: {best_test.get('best_epoch', 'N/A')}")
    print(f"  Test Top-1: {best_test.get('test_top1', 'N/A')}%")
    print(f"  Test Top-5: {best_test.get('test_top5', 'N/A')}%")
    print(f"  Batch size: {best_test.get('batch_size', 'N/A')}")
    print(f"  LR: {best_test.get('lr', 'N/A')}")
    print(f"  Date: {best_test.get('timestamp', 'N/A')[:10]}")
    
    # By subject
    by_subject = defaultdict(list)
    for r in results:
        subject = r.get('subject', 'unknown')
        by_subject[subject].append(safe_float(r.get('test_top1', 0)))
    
    print(f"\n📊 RESULTS BY SUBJECT:")
    for subject in sorted(by_subject.keys()):
        scores = by_subject[subject]
        avg = sum(scores) / len(scores)
        best = max(scores)
        print(f"  Subject {subject}: {len(scores)} runs, "
              f"avg {avg:.2f}%, best {best:.2f}%")
    
    # By batch size
    by_batch = defaultdict(list)
    for r in results:
        batch = r.get('batch_size', 'unknown')
        by_batch[batch].append(safe_float(r.get('test_top1', 0)))
    
    print(f"\n🎯 RESULTS BY BATCH SIZE:")
    for batch in sorted(by_batch.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        scores = by_batch[batch]
        avg = sum(scores) / len(scores)
        best = max(scores)
        print(f"  Batch {batch}: {len(scores)} runs, "
              f"avg {avg:.2f}%, best {best:.2f}%")
    
    # By architecture
    by_arch = defaultdict(list)
    for r in results:
        arch = r.get('architecture', 'unknown')
        by_arch[arch].append(safe_float(r.get('test_top1', 0)))
    
    print(f"\n🏗️  RESULTS BY ARCHITECTURE:")
    for arch in sorted(by_arch.keys()):
        scores = by_arch[arch]
        avg = sum(scores) / len(scores)
        best = max(scores)
        print(f"  {arch}: {len(scores)} runs, "
              f"avg {avg:.2f}%, best {best:.2f}%")
    
    # Comparison to NICE paper
    nice_baseline = 10.4
    print(f"\n📈 COMPARISON TO NICE PAPER (10.4% Top-1):")
    above_baseline = [r for r in results if safe_float(r.get('test_top1', 0)) >= nice_baseline]
    print(f"  Runs matching/exceeding NICE: {len(above_baseline)}/{len(results)}")
    
    if above_baseline:
        best_score = max([safe_float(r.get('test_top1', 0)) for r in above_baseline])
        print(f"  Best improvement: +{best_score - nice_baseline:.2f}%")
    
    print("\n" + "="*100)


def compare_runs(log_file='checkpoints/training_results.txt', run1_idx=0, run2_idx=1):
    """Compare two specific training runs"""
    
    if not os.path.exists(log_file):
        print(f"No log file found: {log_file}")
        return
    
    results = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                entry = {}
                for pair in line.split():
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        entry[key] = value
                results.append(entry)
    
    if len(results) < max(run1_idx, run2_idx) + 1:
        print(f"Not enough runs in log file")
        return
    
    run1 = results[run1_idx]
    run2 = results[run2_idx]
    
    print("\n" + "="*80)
    print(f"COMPARING RUN #{run1_idx+1} vs RUN #{run2_idx+1}")
    print("="*80)
    
    metrics = ['subject', 'architecture', 'best_epoch', 'test_top1', 'test_top5', 
               'batch_size', 'lr']
    
    for metric in metrics:
        v1 = run1.get(metric, 'N/A')
        v2 = run2.get(metric, 'N/A')
        diff = ''
        
        if metric in ['test_top1', 'test_top5'] and v1 != 'N/A' and v2 != 'N/A':
            d = float(v2) - float(v1)
            diff = f" ({'+' if d > 0 else ''}{d:.2f})"
        
        print(f"{metric:15}: {v1:10} → {v2:10}{diff}")
    
    print("="*80)


if __name__ == "__main__":
    import sys
    
    log_file = 'checkpoints/training_results.txt'
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    
    # Run analysis
    analyze_results(log_file)
    
    # If multiple runs exist, compare latest two
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            num_lines = sum(1 for line in f if line.strip())
        
        if num_lines >= 2:
            print("\n")
            compare_runs(log_file, num_lines-2, num_lines-1)
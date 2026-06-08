"""
Stage 2 training epoch monitor.
Watches the training output file and notifies at each epoch completion.
"""

import re
import time
from datetime import datetime, timedelta

OUTPUT_FILE = (
    r"C:\Users\PRECIS~1\AppData\Local\Temp\claude"
    r"\c--Users-Precision-EEGencoder-New-eeg-encoder-direct-on-things"
    r"\a0cfb7d2-0853-4bb4-956d-978b503e9fe0\tasks\bsgp7h5xt.output"
)

TOTAL_EPOCHS = 5
TOTAL_BATCHES_PER_EPOCH = 3722
CHECK_INTERVAL = 60  # seconds

# Patterns
BATCH_PATTERN = re.compile(r'Batch \[(\d+)/3722\]\s+Loss: ([\d.]+)')
EPOCH_PATTERN = re.compile(r'Train loss: ([\d.]+)\s+Val loss: ([\d.]+)')
EPOCH_HEADER  = re.compile(r'Epoch (\d+)/5')
BEST_PATTERN  = re.compile(r'\*\* NEW BEST\s+val_loss=([\d.]+)')

seen_epochs = set()
epoch_times = []  # track time per epoch for ETA

print(f"[{datetime.now().strftime('%H:%M:%S')}] Training monitor started. "
      f"Watching for epoch completions (1-{TOTAL_EPOCHS}).", flush=True)

while len(seen_epochs) < TOTAL_EPOCHS:
    try:
        with open(OUTPUT_FILE, 'r', errors='ignore') as f:
            content = f.read()

        # Find all completed epochs (lines with "Train loss: X   Val loss: Y")
        epoch_headers = EPOCH_HEADER.findall(content)
        epoch_results = EPOCH_PATTERN.findall(content)
        best_flags    = BEST_PATTERN.findall(content)

        # Also get latest batch for current epoch progress
        batch_matches = BATCH_PATTERN.findall(content)

        for i, (train_loss, val_loss) in enumerate(epoch_results):
            epoch_num = i + 1
            if epoch_num not in seen_epochs:
                seen_epochs.add(epoch_num)
                epoch_times.append(datetime.now())

                # Estimate remaining time
                if len(epoch_times) >= 2:
                    elapsed_per_epoch = (epoch_times[-1] - epoch_times[0]) / len(epoch_times)
                    remaining_epochs = TOTAL_EPOCHS - epoch_num
                    eta = epoch_times[-1] + elapsed_per_epoch * remaining_epochs
                    eta_str = f"{int(elapsed_per_epoch.total_seconds()//3600)}h {int((elapsed_per_epoch.total_seconds()%3600)//60)}m/epoch | finishes ~{eta.strftime('%A %I:%M %p')}"
                elif len(epoch_times) == 1:
                    eta_str = "ETA available after epoch 2"
                else:
                    eta_str = "unknown"

                is_best = len(best_flags) >= epoch_num
                best_tag = "  ** NEW BEST **" if is_best else ""

                now = datetime.now().strftime('%H:%M:%S')
                print(flush=True)
                print("=" * 60, flush=True)
                print(f"  [{now}]  EPOCH {epoch_num}/{TOTAL_EPOCHS} COMPLETE{best_tag}", flush=True)
                print(f"  Train loss : {train_loss}", flush=True)
                print(f"  Val loss   : {val_loss}", flush=True)
                print(f"  ETA        : {eta_str}", flush=True)
                print("=" * 60, flush=True)
                print(flush=True)

        # Progress within current epoch
        if batch_matches:
            last_batch, last_loss = batch_matches[-1]
            current_epoch = len(epoch_results) + 1
            if current_epoch <= TOTAL_EPOCHS:
                pct = int(last_batch) / TOTAL_BATCHES_PER_EPOCH * 100
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Epoch {current_epoch}/5 — batch {last_batch}/3722 ({pct:.1f}%)  "
                      f"loss={last_loss}", flush=True)

        if len(seen_epochs) >= TOTAL_EPOCHS:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All 5 epochs complete. "
                  f"Check checkpoints_stage2/best_projector.pth", flush=True)
            break

    except FileNotFoundError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Output file not found yet...", flush=True)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}", flush=True)

    time.sleep(CHECK_INTERVAL)

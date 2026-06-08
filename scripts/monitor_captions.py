"""
Caption job progress monitor.
Watches the captioning output file and prints a milestone report at 25/50/75/100%.
Run alongside generate_captions_local.py — safe to start before or after.
"""

import re
import time
from datetime import datetime, timedelta

OUTPUT_FILE = (
    r"C:\Users\PRECIS~1\AppData\Local\Temp\claude"
    r"\c--Users-Precision-EEGencoder-New-eeg-encoder-direct-on-things"
    r"\6057a818-434d-4534-bda9-7ffee46b5431\tasks\b7h3ja501.output"
)

TOTAL = 16740
MILESTONES = [25, 50, 75, 100]
CHECK_INTERVAL = 60  # seconds between checks

# tqdm writes lines like:
# Captioning:  25%|###       | 4185/16740 [1:23:45<4:12:00,  5.6s/it]
PATTERN = re.compile(
    r'Captioning:\s+\d+%\|[^|]*\|\s*(\d+)/16740\s+\[(\d+:\d+:\d+)<([^,]+),\s*([\d.]+)s/it\]'
)


def parse_eta_str(raw: str) -> str:
    """Convert tqdm ETA string like '4:12:00' to human-readable."""
    raw = raw.strip()
    parts = raw.split(':')
    try:
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, s = 0, int(parts[0]), int(parts[1])
        else:
            return raw
        td = timedelta(hours=h, minutes=m, seconds=s)
        finish = datetime.now() + td
        return f"{h}h {m}m  (finishes ~{finish.strftime('%I:%M %p')})"
    except Exception:
        return raw


def get_latest_progress(content: str):
    """Return (done, eta_str, speed) from the last tqdm line in the file."""
    matches = PATTERN.findall(content)
    if not matches:
        return None
    done_str, elapsed, eta_raw, speed = matches[-1]
    return int(done_str), parse_eta_str(eta_raw), float(speed)


hit = set()
remaining_milestones = sorted(MILESTONES)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitor started. "
      f"Watching for milestones: {remaining_milestones}", flush=True)

while remaining_milestones:
    try:
        with open(OUTPUT_FILE, 'r', errors='ignore') as f:
            content = f.read()

        result = get_latest_progress(content)
        if result:
            done, eta_str, speed = result
            pct = done / TOTAL * 100

            for m in list(remaining_milestones):
                if pct >= m and m not in hit:
                    hit.add(m)
                    remaining_milestones.remove(m)
                    now = datetime.now().strftime('%H:%M:%S')
                    print(flush=True)
                    print("=" * 60, flush=True)
                    print(f"  [{now}]  MILESTONE: {m}% COMPLETE", flush=True)
                    print(f"  Images done : {done:,} / {TOTAL:,}", flush=True)
                    print(f"  Speed       : {speed:.1f}s/image", flush=True)
                    if m < 100:
                        print(f"  ETA to 100% : {eta_str}", flush=True)
                    else:
                        print("  Captioning COMPLETE. Check things_captions.json", flush=True)
                    print("=" * 60, flush=True)
                    print(flush=True)

    except FileNotFoundError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Output file not found yet — waiting...",
              flush=True)
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}", flush=True)

    if remaining_milestones:
        time.sleep(CHECK_INTERVAL)

print(f"[{datetime.now().strftime('%H:%M:%S')}] All milestones reached. Monitor done.", flush=True)

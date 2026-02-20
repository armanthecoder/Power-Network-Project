"""
Batch Runner — Sweep over sink counts and random seeds
=======================================================
Outer loop: NUM_SINKS  = [2, 4, 6, 8, 10, 12, 15]
Inner loop: RANDOM_SEED = [2, 3, 4, 5, 6]

For each combination:
  1. Patches constants_paper_version.py (NUM_SINKS, RANDOM_SEED, VISUALIZE_ON=False)
  2. Runs main_paper_version.py as a subprocess
  3. Moves the output folder into a master test directory
  4. Skips if the run exceeds 45 minutes

Usage:
    python batch_runner.py
"""

import os
import re
import sys
import time
import shutil
import subprocess

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
SINK_COUNTS   = [2,4, 6, 8, 10, 12, 15,20,30,50]
RANDOM_SEEDS  = [2, 3, 4, 5, 6, 7, 8]
   # 45 minutes per run

CONSTANTS_FILE = "constants_paper_version.py"
MAIN_FILE      = "main_paper_version.py"
WORK_DIR       = os.path.dirname(os.path.abspath(__file__))

# Master output directory name
MASTER_DIR = f"Testfifth_second_part_{SINK_COUNTS[0]}-{SINK_COUNTS[-1]}sinks_{RANDOM_SEEDS[0]}-{RANDOM_SEEDS[-1]}seeds"


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _patch_constant(text, name, new_value):
    """Replace  NAME = <old>  with  NAME = <new>  in the constants file text."""
    # Match patterns like:  NUM_SINKS =4   or   RANDOM_SEED = 31
    pattern = rf"^({name}\s*=\s*).*$"
    replacement = rf"\g<1>{new_value}"
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n == 0:
        raise ValueError(f"Could not find '{name}' in constants file.")
    return new_text


def patch_constants(num_sinks, random_seed):
    """Rewrite constants_paper_version.py with the given parameters."""
    path = os.path.join(WORK_DIR, CONSTANTS_FILE)
    with open(path, "r") as f:
        text = f.read()

    text = _patch_constant(text, "NUM_SINKS",    num_sinks)
    text = _patch_constant(text, "RANDOM_SEED",  random_seed)
    text = _patch_constant(text, "VISUALIZE_ON", "False")

    with open(path, "w") as f:
        f.write(text)


def restore_constants(original_text):
    """Restore the original constants file."""
    path = os.path.join(WORK_DIR, CONSTANTS_FILE)
    with open(path, "w") as f:
        f.write(original_text)


def find_new_output_folder(existing_folders):
    """Return the first new folder in WORK_DIR that wasn't there before."""
    current = set(
        e for e in os.listdir(WORK_DIR)
        if os.path.isdir(os.path.join(WORK_DIR, e))
    )
    new = current - existing_folders
    # Filter to folders matching the expected pattern
    candidates = sorted(
        d for d in new if "sinks_" in d and "seed" in d
    )
    return candidates[0] if candidates else None


# ─────────────────────────────────────────────────────────────
#  MAIN BATCH LOOP
# ─────────────────────────────────────────────────────────────

def main():
    os.chdir(WORK_DIR)

    # Save original constants so we can restore at the end
    const_path = os.path.join(WORK_DIR, CONSTANTS_FILE)
    with open(const_path, "r") as f:
        original_constants = f.read()

    # Create master output directory
    master_path = os.path.join(WORK_DIR, MASTER_DIR)
    os.makedirs(master_path, exist_ok=True)

    total = len(SINK_COUNTS) * len(RANDOM_SEEDS)
    done, skipped, failed = 0, 0, 0
    log_lines = []

    print("=" * 70)
    print(f"  BATCH RUNNER — {total} combinations")
    print(f"  Sinks:  {SINK_COUNTS}")
    print(f"  Seeds:  {RANDOM_SEEDS}")
    print(f"  Timeout: adaptive (4 * num_sinks * 60 sec)")
    print(f"  Output:  {MASTER_DIR}/")
    print("=" * 70)

    try:
        for n_sinks in SINK_COUNTS:
            for seed in RANDOM_SEEDS:
                label = f"sinks={n_sinks:>2}, seed={seed}"
                print(f"\n{'─' * 70}")
                print(f"  [{done + skipped + failed + 1}/{total}]  {label}")
                print(f"{'─' * 70}")

                # Snapshot existing folders before the run
                existing_folders = set(
                    e for e in os.listdir(WORK_DIR)
                    if os.path.isdir(os.path.join(WORK_DIR, e))
                )

                # Patch constants
                patch_constants(n_sinks, seed)

                # Adaptive timeout: 4 * n_sinks * 60 seconds
                timeout_sec = 4 * n_sinks * 60

                # Run simulation as subprocess
                t0 = time.time()
                try:
                    result = subprocess.run(
                        [sys.executable, MAIN_FILE],
                        cwd=WORK_DIR,
                        timeout=timeout_sec,
                        capture_output=True,
                        text=True,
                    )
                    elapsed = time.time() - t0
                    ret = result.returncode

                    if ret != 0:
                        msg = f"  FAILED (exit code {ret}) after {elapsed:.1f}s"
                        print(msg)
                        print(f"  stderr: {result.stderr[-500:]}" if result.stderr else "")
                        log_lines.append(f"FAIL  {label}  exit={ret}  {elapsed:.1f}s")
                        failed += 1
                    else:
                        msg = f"  OK  ({elapsed:.1f}s)"
                        print(msg)
                        log_lines.append(f"OK    {label}  {elapsed:.1f}s")
                        done += 1

                except subprocess.TimeoutExpired:
                    elapsed = time.time() - t0
                    msg = f"  TIMEOUT ({elapsed:.1f}s > {TIMEOUT_SEC}s) — skipping"
                    print(msg)
                    log_lines.append(f"SKIP  {label}  timeout {elapsed:.1f}s")
                    skipped += 1

                # Move the output folder into master directory
                new_folder = find_new_output_folder(existing_folders)
                if new_folder:
                    src_path = os.path.join(WORK_DIR, new_folder)
                    dst_path = os.path.join(master_path, new_folder)
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.move(src_path, dst_path)
                    print(f"  Moved: {new_folder}  ->  {MASTER_DIR}/")
                else:
                    print("  (no output folder found)")

    except KeyboardInterrupt:
        print("\n\n  Batch run interrupted by user (Ctrl+C)")

    finally:
        # Always restore original constants
        restore_constants(original_constants)
        print("\n  Constants file restored to original values.")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  BATCH COMPLETE")
    print(f"    OK:      {done}")
    print(f"    Skipped: {skipped}  (timeout)")
    print(f"    Failed:  {failed}")
    print(f"    Output:  {MASTER_DIR}/")
    print(f"{'=' * 70}")

    # Save log
    log_path = os.path.join(master_path, "batch_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Sinks:  {SINK_COUNTS}\n")
        f.write(f"Seeds:  {RANDOM_SEEDS}\n")
        f.write(f"Timeout: {TIMEOUT_SEC}s\n\n")
        f.write("\n".join(log_lines) + "\n")
    print(f"  Log saved: {log_path}")


if __name__ == "__main__":
    main()

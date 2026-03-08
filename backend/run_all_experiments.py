"""Sequential experiment orchestrator.

Runs all Protocol-2 experiments one at a time on GPU.
P2 rerun is already running; this script waits for it then queues the rest.

Usage:
    python run_all_experiments.py             # waits for P2 then runs the rest
    python run_all_experiments.py --skip N    # skip first N experiments
    python run_all_experiments.py --start EXP # start from a named experiment
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

BACKEND = Path(__file__).parent
DATA    = BACKEND.parent / "data"

# Ordered queue — (script, data_dir, expected_runs, description)
QUEUE = [
    ("run_p2_rerun.py",            "p2_rerun",            45,  "P2 cost-sensitivity rerun"),
    ("run_p5_rerun.py",            "p5_rerun",            30,  "P5 coordination advantage rerun"),
    ("run_exp3_metastability.py",  "exp3_metastability",  45,  "Exp 3 metastability (square-wave)"),
    ("run_exp11_saturation.py",    "exp11_saturation",    60,  "Exp 11 saturation forcing"),
    ("run_exp12_irreversibility.py","exp12_irreversibility",15, "Exp 12 irreversibility (3-phase)"),
    ("run_exp4_noise.py",          "exp4_noise",          120, "Exp 4 noise at boundary"),
    ("run_exp7_topology.py",       "exp7_topology",       45,  "Exp 7 topology (adjacency)"),
    ("run_exp8_memory_depth.py",   "exp8_memory_depth",   75,  "Exp 8 memory depth scaling"),
    ("run_exp9_observability.py",  "exp9_observability",  75,  "Exp 9 partial observability"),
    ("run_exp13_redistribution.py","exp13_redistribution",75,  "Exp 13 redistribution"),
    ("run_exp14_coupling_window.py","exp14_coupling_window",135,"Exp 14 coupling window"),
    ("run_exp15_ablation.py",      "exp15_ablation",      75,  "Exp 15 protocol ablation"),
    ("run_exp1_tax_sweep.py",      "exp1_tax_sweep",      150, "Exp 1 fine tax sweep"),
]

WORKERS = 12
LOG = BACKEND / "orchestrator.log"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_manifests(data_dir: str) -> int:
    d = DATA / data_dir
    if not d.exists():
        return 0
    return sum(1 for _ in d.rglob("manifest.json"))


def wait_for_running_p2() -> None:
    """Block until the already-running P2 process finishes."""
    import psutil
    log("Waiting for already-running P2 rerun to complete...")
    while True:
        running = any(
            "run_p2_rerun.py" in " ".join(p.cmdline())
            for p in psutil.process_iter(["cmdline"])
            if p.info["cmdline"]
        )
        if not running:
            n = count_manifests("p2_rerun")
            log(f"P2 rerun finished — {n} manifests collected.")
            return
        n = count_manifests("p2_rerun")
        log(f"  P2 still running... {n} manifests so far")
        time.sleep(60)


def run_experiment(script: str, data_dir: str, expected: int, desc: str) -> bool:
    """Run one experiment and return True on success."""
    n_before = count_manifests(data_dir)
    log(f"{'='*60}")
    log(f"STARTING: {desc}")
    log(f"  Script  : {script}")
    log(f"  Output  : data/{data_dir}/")
    log(f"  Expected: {expected} runs  Workers: {WORKERS}")
    log(f"{'='*60}")

    cmd = [sys.executable, str(BACKEND / script), "--workers", str(WORKERS)]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BACKEND),
            capture_output=False,
            text=True,
        )
    except Exception as e:
        log(f"ERROR launching {script}: {e}")
        return False

    elapsed = time.time() - t0
    n_after = count_manifests(data_dir)
    new_runs = n_after - n_before

    if result.returncode != 0:
        log(f"FAILED: {desc} (exit {result.returncode}) — {new_runs} runs in {elapsed/60:.1f} min")
        return False

    log(f"DONE: {desc} — {new_runs}/{expected} runs in {elapsed/60:.1f} min")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip",  type=int, default=0,   help="Skip first N queue entries")
    parser.add_argument("--start", type=str, default=None, help="Start from named script")
    args = parser.parse_args()

    queue = QUEUE[args.skip:]
    if args.start:
        queue = [(s, d, e, desc) for s, d, e, desc in queue if args.start in s]
        if not queue:
            print(f"No match for --start {args.start}")
            sys.exit(1)

    log("=== Experiment Orchestrator Started ===")
    log(f"Queue: {len(queue)} experiments to run")
    for i, (s, d, e, desc) in enumerate(queue):
        log(f"  {i+1:2d}. {desc} ({e} runs)")

    # First entry is P2 rerun — run it inline like the rest
    pass

    total_ok = 0
    total_fail = 0
    for script, data_dir, expected, desc in queue:
        ok = run_experiment(script, data_dir, expected, desc)
        if ok:
            total_ok += 1
        else:
            total_fail += 1

    log(f"=== Orchestrator Done: {total_ok} OK, {total_fail} failed ===")


if __name__ == "__main__":
    main()

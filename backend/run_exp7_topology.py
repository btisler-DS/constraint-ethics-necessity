"""Experiment 7 — Adjacency vs abstraction propagation (communication topology).

Tests whether restricting communication to adjacent pairs (linear chain A-B-C)
vs. allowing full broadcast (all-to-all) produces qualitatively different
crystallization dynamics, QRC, and protocol structure.

Theory prediction
-----------------
  If true protocol structure emerges from LOCAL adjacency constraints, then
  restricting to a linear chain should still produce crystallization, possibly
  with different dynamics. If global broadcast causes "instant coherence"
  (suggesting substitution/globalization does the work), then the chain topology
  should show delayed or failed crystallization.

Topologies tested
-----------------
  broadcast   — all agents see all others (default / baseline)
                  A ↔ B ↔ C ↔ A (all-to-all)

  linear_chain — linear adjacency only (A-B-C)
                  A → B → C (A sees B only, B sees A+C, C sees B only)
                  B is the middle node that propagates information.

  star        — Agent C is hub (A and B only communicate via C)
                  A ↔ C, B ↔ C, A does NOT see B directly

Implementation
--------------
  comm_buffer.receive_all is patched to zero out blocked-sender channels
  while preserving the tensor shape expected by the agent networks.

  For a 3-agent system with signal_dim=8:
    receive_all returns shape (16,) = concat of 2 × 8-dim signals
  The two slots are filled with:
    signals[name_a] and signals[name_b] (sorted alphabetically)
  Blocked senders get zeros in their slot.

  Conditions: broadcast × linear_chain × star × 3 = 3 × 15 seeds = 45 runs.

Output
------
  data/exp7_topology/seed_{s:02d}_{condition}/manifest.json
  data/exp7_topology/exp7_summary.json

Usage
-----
  python run_exp7_topology.py
  python run_exp7_topology.py --workers 4
  python run_exp7_topology.py --dry-run
  python run_exp7_topology.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SEEDS        = list(range(15))
EPOCHS       = 500
EPISODES     = 10
DECLARE_COST = 1.0
QUERY_COST   = 1.5
RESPOND_COST = 0.8

# Adjacency maps: agent_name → set of agents it is ALLOWED to receive from
# (broadcast = all others, i.e., no restriction)
TOPOLOGIES = {
    "broadcast":    {"A": {"B", "C"}, "B": {"A", "C"}, "C": {"A", "B"}},
    "linear_chain": {"A": {"B"},      "B": {"A", "C"}, "C": {"B"}},
    "star":         {"A": {"C"},      "B": {"C"},      "C": {"A", "B"}},
}

CONDITIONS = [
    {"label": "broadcast",    "topology": "broadcast"},
    {"label": "linear_chain", "topology": "linear_chain"},
    {"label": "star",         "topology": "star"},
]

DATA_ROOT = Path(__file__).parent.parent / "data" / "exp7_topology"


# ── Worker ─────────────────────────────────────────────────────────────────────

def _run_one(spec: dict) -> dict:
    """Run one seed under one topology condition. Called in subprocess."""
    import sys, os
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from simulation.engine import SimulationEngine, SimulationConfig

    topology_map = spec["topology_map"]   # passed as dict

    config = SimulationConfig(
        seed=spec["seed"],
        num_epochs=EPOCHS,
        episodes_per_epoch=EPISODES,
        protocol=1,
        declare_cost=DECLARE_COST,
        query_cost=QUERY_COST,
        respond_cost=RESPOND_COST,
        output_dir=spec["output_dir"],
    )
    engine = SimulationEngine(config=config)

    if spec["condition"] != "broadcast":
        # Patch receive_all to zero out blocked-sender slots while preserving
        # the expected tensor shape (signal_dim * (num_agents - 1) = 16).
        original_receive     = engine.comm_buffer.receive
        original_receive_all = engine.comm_buffer.receive_all
        signal_dim           = engine.comm_buffer.signal_dim   # 8

        def topology_receive_all(agent_name: str) -> torch.Tensor:
            allowed = topology_map.get(agent_name, set())
            per_agent = original_receive(agent_name)
            # Sort sender names for deterministic slot order
            result = []
            for other_name in sorted(per_agent.keys()):
                sig = per_agent[other_name]
                if other_name in allowed:
                    result.append(sig)
                else:
                    result.append(torch.zeros_like(sig))
            if not result:
                return torch.zeros(signal_dim * 2)
            return torch.cat(result)

        engine.comm_buffer.receive_all = topology_receive_all

    engine.run()
    return {"seed": spec["seed"], "condition": spec["label"]}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_manifest(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _is_crystallized(m: dict) -> bool | None:
    val = m.get("crystallized")
    if val is not None:
        return bool(val)
    fm = m.get("final_metrics", {})
    val = fm.get("crystallized")
    if val is not None:
        return bool(val)
    crys_epoch = m.get("crystallization_epoch")
    if crys_epoch is not None:
        return True
    if "crystallization_epoch" in m:
        return False
    return None


def _extract(m: dict) -> dict:
    fm = m.get("final_metrics", m)
    return {
        "crystallized":   _is_crystallized(m),
        "crys_epoch":     m.get("crystallization_epoch"),
        "type_entropy":   fm.get("type_entropy"),
        "qrc":            fm.get("query_response_coupling") or fm.get("qrc"),
        "query_rate":     fm.get("query_rate"),
        "survival_rate":  fm.get("survival_rate"),
    }


# ── Campaign runner ─────────────────────────────────────────────────────────────

def run_campaign(seeds: list[int], workers: int, dry_run: bool) -> None:
    jobs = []
    for cond in CONDITIONS:
        for seed in seeds:
            out_dir = DATA_ROOT / f"seed_{seed:02d}_{cond['label']}"
            if (out_dir / "manifest.json").exists():
                continue
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
            jobs.append({
                "seed":         seed,
                "condition":    cond["label"],
                "topology_map": TOPOLOGIES[cond["topology"]],
                "output_dir":   str(out_dir),
            })

    if not jobs:
        print("  No new jobs (all data present).")
        return

    print(f"\n  Running {len(jobs)} jobs  (workers={workers})")
    if dry_run:
        for j in jobs[:5]:
            print(f"  DRY-RUN: {j['condition']}  seed={j['seed']}")
        if len(jobs) > 5:
            print(f"  ... ({len(jobs)-5} more)")
        return

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                done += 1
                print(f"  [{done:3d}/{len(jobs)}]  {r['condition']}  seed={r['seed']:02d}")
            except Exception as e:
                j = futures[fut]
                print(f"  ERROR: {j['condition']}  seed={j['seed']}: {e}")
    print(f"  Done in {time.time()-t0:.0f}s")


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze(seeds: list[int]) -> dict:
    per_condition: dict[str, list[dict]] = {c["label"]: [] for c in CONDITIONS}

    for cond in CONDITIONS:
        for seed in seeds:
            out_dir = DATA_ROOT / f"seed_{seed:02d}_{cond['label']}"
            m = _load_manifest(out_dir / "manifest.json")
            if m is None:
                continue
            d = _extract(m)
            d["seed"] = seed
            d["condition"] = cond["label"]
            per_condition[cond["label"]].append(d)

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    condition_stats = {}
    for cond in CONDITIONS:
        label  = cond["label"]
        rows   = per_condition[label]
        n      = len(rows)
        n_crys = sum(1 for r in rows if r.get("crystallized"))
        condition_stats[label] = {
            "topology":             cond["topology"],
            "n":                    n,
            "n_crystallized":       n_crys,
            "crystallization_rate": round(n_crys / n, 4) if n else None,
            "mean_qrc":             _mean([r["qrc"] for r in rows]),
            "mean_type_entropy":    _mean([r["type_entropy"] for r in rows]),
            "mean_query_rate":      _mean([r["query_rate"] for r in rows]),
            "mean_survival_rate":   _mean([r["survival_rate"] for r in rows]),
            "mean_crys_epoch":      _mean([r["crys_epoch"] for r in rows
                                           if r.get("crystallized") and r["crys_epoch"]]),
        }

    # Does topology restriction degrade performance relative to broadcast?
    broadcast_rate = condition_stats["broadcast"]["crystallization_rate"]
    broadcast_qrc  = condition_stats["broadcast"]["mean_qrc"]

    topology_effects = {}
    for cond in CONDITIONS[1:]:
        label = cond["label"]
        cs = condition_stats[label]
        rate = cs["crystallization_rate"]
        qrc  = cs["mean_qrc"]
        topology_effects[label] = {
            "crys_rate_vs_broadcast": round(rate / broadcast_rate, 4)
                                      if (rate and broadcast_rate) else None,
            "qrc_vs_broadcast":       round(qrc / broadcast_qrc, 4)
                                      if (qrc  and broadcast_qrc)  else None,
        }

    return {
        "experiment":       "7",
        "description":      "Adjacency vs abstraction propagation (topology)",
        "generated":        datetime.now().isoformat(),
        "n_seeds":          len(seeds),
        "topologies":       {k: list(v.items()) for k, v in TOPOLOGIES.items()},
        "conditions":       condition_stats,
        "topology_effects": topology_effects,
    }


# ── Summary writer ─────────────────────────────────────────────────────────────

def write_summary(result: dict, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nExp 7 summary: {out_path}")

    print(f"\n  {'Condition':<14}  {'n':>3}  {'rate':>5}  "
          f"{'QRC':>6}  {'H':>6}  {'Qrate':>6}  {'crys_epoch':>10}")
    for cond in CONDITIONS:
        label = cond["label"]
        cs    = result["conditions"][label]
        rate  = f"{cs['crystallization_rate']:.2f}" if cs["crystallization_rate"] is not None else " N/A"
        qrc   = f"{cs['mean_qrc']:.3f}"             if cs["mean_qrc"]             is not None else " N/A"
        H     = f"{cs['mean_type_entropy']:.3f}"    if cs["mean_type_entropy"]    is not None else " N/A"
        qr    = f"{cs['mean_query_rate']:.3f}"       if cs["mean_query_rate"]       is not None else " N/A"
        ep    = f"{cs['mean_crys_epoch']:.0f}"      if cs["mean_crys_epoch"]      is not None else " N/A"
        print(f"  {label:<14}  {cs['n']:>3}  {rate:>5}  {qrc:>6}  {H:>6}  {qr:>6}  {ep:>10}")

    if result["topology_effects"]:
        print(f"\n  Topology effects relative to broadcast:")
        for label, eff in result["topology_effects"].items():
            rr = f"{eff['crys_rate_vs_broadcast']:.2f}x" if eff["crys_rate_vs_broadcast"] else " N/A"
            qr = f"{eff['qrc_vs_broadcast']:.2f}x"       if eff["qrc_vs_broadcast"]       else " N/A"
            print(f"  {label}: crys_rate={rr}  QRC={qr}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 7 -- Adjacency vs abstraction propagation (topology)")
    parser.add_argument("--seeds",        type=int, nargs=2, default=[0, 15],
                        metavar=("START", "END"))
    parser.add_argument("--workers",      type=int, default=None)
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    seeds   = list(range(args.seeds[0], args.seeds[1]))
    workers = args.workers or max(1, (os.cpu_count() or 4) - 1)

    print(f"Exp 7 -- Topology  --  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}  Workers: {workers}")
    print(f"Conditions: {len(CONDITIONS)}  x  {len(seeds)} seeds = {len(CONDITIONS)*len(seeds)} runs")
    print(f"Data: {DATA_ROOT}")

    if args.analyze_only:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        result = analyze(seeds)
        write_summary(result, DATA_ROOT / "exp7_summary.json")
        return

    if not args.dry_run:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

    run_campaign(seeds, workers, args.dry_run)

    if not args.dry_run:
        result = analyze(seeds)
        write_summary(result, DATA_ROOT / "exp7_summary.json")

    print("\nDone.")


if __name__ == "__main__":
    main()

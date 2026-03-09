"""Protocol 2 — Control condition: all agents unconstrained.

No ethical tax applied. Agents operate under baseline metabolic accounting
only (signal cost, move cost, survival bonus). Without the pressure gradient
from ethical constraint, agents are predicted to collapse into exploitation
loops and cease generating novel questions.

Prediction: Unconstrained agents develop high exploitation_loop_rate and
low sustained_structure_score (H1). Effect size vs constrained condition
predicted to be d > 0.8 (H3).

Run parameters are preregistration-locked. Do not modify before registration.
"""

from simulation.engine import SimulationEngine, SimulationConfig

config = SimulationConfig(
    protocol=2,
    population_mode="all_unconstrained",
    seed=42,
    num_epochs=100,
    episodes_per_epoch=10,
    output_dir="data/p2_unconstrained",
)

if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else config.seed
    num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    def print_epoch(m: dict) -> None:
        inq = m.get("inquiry", {})
        te_str = f" | H: {inq['type_entropy']:.3f}" if isinstance(inq, dict) and "type_entropy" in inq else ""
        coll = m.get("collapse_metrics", {}).get("interrogative_collapse", {})
        loop = m.get("collapse_metrics", {}).get("exploitation_loop", {})
        collapse_str = " [COLLAPSE]" if coll.get("collapse_detected") else ""
        loop_str = " [LOOP]" if loop.get("loop_detected") else ""
        print(
            f"Epoch {m['epoch']:3d} [unconstrained] | "
            f"Target: {m['target_reached_rate']:.2f}{te_str}{collapse_str}{loop_str}"
        )

    run_config = SimulationConfig(
        protocol=2,
        population_mode="all_unconstrained",
        seed=seed,
        num_epochs=num_epochs,
        episodes_per_epoch=10,
        output_dir=f"data/p2_unconstrained/seed_{seed}",
    )
    engine = SimulationEngine(config=run_config, epoch_callback=print_epoch)
    results = engine.run()
    print(f"Run complete. {len(results)} epochs logged.")

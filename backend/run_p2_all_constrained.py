"""Protocol 2 — Experimental condition: all agents constrained.

Ethical tax applied to all agents as a Landauer-style resource cost on
exploitation loops. Agents under ethical constraint must pay an additional
cost (ETHICAL_TAX_RATE × signal_cost) whenever they have been in a
non-query state for 3+ consecutive steps (the Omega→Delta→Phi threshold).

Prediction: Constrained agents maintain interrogative diversity — sustained
type entropy and query-response coupling through final epochs (H2).

Run parameters are preregistration-locked. Do not modify before registration.
"""

from simulation.engine import SimulationEngine, SimulationConfig

config = SimulationConfig(
    protocol=2,
    population_mode="all_constrained",
    seed=42,
    num_epochs=100,
    episodes_per_epoch=10,
    output_dir="data/p2_constrained",
)

if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else config.seed
    num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    def print_epoch(m: dict) -> None:
        inq = m.get("inquiry", {})
        ec = m.get("ethical_constraint", {})
        total_ethical = sum(ec.get("ethical_cost_by_agent", {}).values())
        te_str = f" | H: {inq['type_entropy']:.3f}" if isinstance(inq, dict) and "type_entropy" in inq else ""
        coll = m.get("collapse_metrics", {}).get("interrogative_collapse", {})
        collapse_str = " [COLLAPSE]" if coll.get("collapse_detected") else ""
        print(
            f"Epoch {m['epoch']:3d} [constrained] | "
            f"Target: {m['target_reached_rate']:.2f} | "
            f"Ethical cost: {total_ethical:.4f}"
            f"{te_str}{collapse_str}"
        )

    run_config = SimulationConfig(
        protocol=2,
        population_mode="all_constrained",
        seed=seed,
        num_epochs=num_epochs,
        episodes_per_epoch=10,
        output_dir=f"data/p2_constrained/seed_{seed}",
    )
    engine = SimulationEngine(config=run_config, epoch_callback=print_epoch)
    results = engine.run()
    print(f"Run complete. {len(results)} epochs logged.")

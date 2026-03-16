"""Simulation engine orchestrator.

Wires together environment, agents, communication buffer, training,
and metrics collection. Runs episodes and epochs, producing metrics
snapshots at each epoch boundary.

Protocol selection is via SimulationConfig.protocol:
    0 — Baseline (flat Landauer tax, no type head gradient — replicates Run 10)
    1 — Interrogative Emergence (Gumbel-Softmax, differential costs)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical, Normal

from .environment import Environment, EnvironmentConfig
from .comm_buffer import CommBuffer, CommBufferConfig
from .agents import AgentA, AgentB, AgentC
from .training.reinforce import update_agents
from .metrics.shannon_entropy import compute_per_agent_entropy
from .metrics.mutual_information import compute_pairwise_mi
from .metrics.transfer_entropy import compute_all_pairs_te
from .metrics.zipf_analysis import compute_zipf_per_agent
from .metrics.energy_roi import compute_energy_roi
from .metrics.pca_snapshot import collect_signal_samples, fit_pca_and_project
from .protocols import create_protocol, Protocol2
from .metrics.collapse_metrics import interrogative_collapse_rate, exploitation_loop_detection
from .utils.seeding import set_all_seeds

# Protocol 4 — Sacrifice-Conflict scenario constants
_CRITICAL_ENERGY_THRESHOLD: float = 20.0  # trigger token spawn when agent drops below this
_SACRIFICE_WINDOW: int = 10               # steps after trigger to classify sacrifice choice


@dataclass
class SimulationConfig:
    seed: int = 42
    num_epochs: int = 100
    episodes_per_epoch: int = 10
    grid_size: int = 20
    num_obstacles: int = 8
    z_layers: int = 8
    max_steps: int = 100
    energy_budget: float = 100.0
    move_cost: float = 1.0
    collision_penalty: float = 5.0
    signal_dim: int = 8
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    communication_tax_rate: float = 0.01
    survival_bonus: float = 0.1
    protocol: int = 1      # 0=Baseline, 1=Interrogative Emergence, 2=Ethical Constraints
    declare_cost: float = 1.0   # DECLARE signal cost multiplier (Protocol 1)
    query_cost: float = 1.5     # QUERY signal cost multiplier (Protocol 1)
    respond_cost: float = 0.8   # RESPOND signal cost multiplier (Protocol 1)
    # Protocol 2: Ethical Constraints — fixed constants, two conditions only
    population_mode: str = "all_unconstrained"  # "all_unconstrained" | "all_constrained"
    output_dir: str = "."       # Directory for manifest and report output
    training_frozen: bool = False          # If True, skip optimizer step (phase 2 of hysteresis)
    ablate_agent_c_type_head: bool = False # Experiment 6: freeze Agent C type_head only
    # Counter-wave discrimination (Experiments 3–5)
    counter_wave_mode: int = 0             # 0=normal, 3=no-boundary, 4=no-bonus, 5=pressure-hold
    post_success_steps: int = 20           # Exp 3: steps to continue after success event
    post_success_pressure_episodes: int = 3  # Exp 5: episodes at elevated tax after success
    post_success_tax_multiplier: float = 3.0 # Exp 5: tax rate multiplier during pressure hold
    device: str = "cpu"                    # Compute device: "cpu", "cuda:0", "cuda:1", ...
    # Protocol 4: Depth control and ablation
    depth: int = 1                         # 0=feedforward baseline, 1=full arch, 2=+self_model_gru
    ablate_self_model_inputs: bool = False # Depth 2 only: zero self_model_gru inputs for ablation
    freeze_self_model_gru: bool = False    # Boundary condition: depth=2 arch, self_model_gru frozen at random init
    # Protocol 5: Welfare coupling and proportional energy scaling
    welfare_coupled: bool = False          # If True, R_i = 0.5*self_reward + 0.5*mean(other_agent_rewards)
    critical_energy_threshold: float = 20.0  # Sacrifice-Conflict trigger; scale with energy_budget


class SimulationEngine:
    """Main simulation orchestrator."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        epoch_callback: Callable[[dict], None] | None = None,
    ):
        self.config = config or SimulationConfig()
        self.epoch_callback = epoch_callback
        self._paused = False
        self._stopped = False

        # Initialize seeds
        set_all_seeds(self.config.seed)

        # Protocol manager — drives reward, type resolution, and epoch extras
        self.protocol = create_protocol(
            self.config.protocol,
            declare_cost=self.config.declare_cost,
            query_cost=self.config.query_cost,
            respond_cost=self.config.respond_cost,
            population_mode=self.config.population_mode,
        )

        # Environment
        env_config = EnvironmentConfig(
            grid_size=self.config.grid_size,
            num_obstacles=self.config.num_obstacles,
            z_layers=self.config.z_layers,
            max_steps=self.config.max_steps,
            energy_budget=self.config.energy_budget,
            move_cost=self.config.move_cost,
            collision_penalty=self.config.collision_penalty,
        )
        self.env = Environment(env_config)

        # Communication buffer
        comm_config = CommBufferConfig(signal_dim=self.config.signal_dim)
        self.comm_buffer = CommBuffer(comm_config)

        # Build agents
        # Agent A obs_dim: 1 (target) + num_obstacles + 2 (other agents) + 1 (token distance)
        obs_dim_a = 1 + self.config.num_obstacles + 2 + 1  # 12 with default num_obstacles=8
        self.agent_a = AgentA(
            obs_dim=obs_dim_a,
            signal_dim=self.config.signal_dim,
            hidden_dim=self.config.hidden_dim,
            depth=self.config.depth,
            ablate_self_model_inputs=self.config.ablate_self_model_inputs,
            freeze_self_model_gru=self.config.freeze_self_model_gru,
        )

        self.agent_b = AgentB(
            signal_dim=self.config.signal_dim,
            hidden_dim=self.config.hidden_dim,
            depth=self.config.depth,
        )

        # Agent C obs_dim: pairwise distances — token added as 13th entity
        # 13 entities: target + obstacles + 3 agents + token → 13*12//2 = 78 pairs
        n_entities = 1 + self.config.num_obstacles + 3 + 1  # token is the +1
        obs_dim_c = n_entities * (n_entities - 1) // 2      # 78 with default num_obstacles=8
        self.agent_c = AgentC(
            obs_dim=obs_dim_c,
            signal_dim=self.config.signal_dim,
            hidden_dim=self.config.hidden_dim,
            depth=self.config.depth,
        )

        # Token state — managed entirely in engine (engine overlay architecture)
        # Populated by Step 6 Sacrifice-Conflict implementation.
        self._token_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self._token_active: bool = False

        self.agents = [self.agent_a, self.agent_b, self.agent_c]

        # Move all agent parameters to target device
        self.device = torch.device(self.config.device)
        for agent in self.agents:
            agent.to(self.device)

        # Freeze type_head for Protocol 0 — no gradient through type classification
        if not self.protocol.should_train_type_head():
            for agent in self.agents:
                agent.freeze_type_head()

        # Experiment 6: ablate Agent C's type_head only (A and B remain trainable)
        if self.config.ablate_agent_c_type_head and self.protocol.should_train_type_head():
            self.agent_c.freeze_type_head()

        # Optimizers
        self.optimizers = [
            optim.Adam(agent.parameters(), lr=self.config.learning_rate)
            for agent in self.agents
        ]

        # Metrics history
        self.epoch_metrics: list[dict] = []
        self._all_signal_samples: list[list[dict]] = []

    def run(self) -> list[dict]:
        """Run the full simulation. Returns list of epoch metrics."""
        self._all_signal_samples = []

        for epoch in range(self.config.num_epochs):
            if self._stopped:
                break

            while self._paused:
                time.sleep(0.1)
                if self._stopped:
                    break

            metrics = self._run_epoch(epoch)
            self.epoch_metrics.append(metrics)

            if self.epoch_callback:
                self.epoch_callback(metrics)

        # Post-run: fit global PCA and inject per-epoch projections
        if self._all_signal_samples:
            projections = fit_pca_and_project(self._all_signal_samples)
            for i, proj in enumerate(projections):
                if i < len(self.epoch_metrics):
                    self.epoch_metrics[i]['signal_pca'] = proj

        # Write run manifest and per-epoch series (for confirmatory analysis)
        self._write_manifest("manifest.json")
        self._write_epoch_series("epoch_series.json")

        return self.epoch_metrics

    def _augment_obs_with_token(self, obs: dict) -> dict:
        """Append token features to all three agent observations.

        Called after every env.reset() and env.step() to extend observations
        from their base dims to the token-aware dims:
          AgentA: 11-dim → 12-dim  (append scalar distance to token)
          AgentB: (1,Z,H,W) → (2,Z,H,W)  (stack binary token presence channel)
          AgentC: 66-dim → 78-dim  (append 12 pairwise distances: token↔each entity)

        When token is inactive all appended features are 0.0 / zero-channel.
        """
        if self._token_active:
            token_pos = self._token_pos.astype(np.float32)
        else:
            token_pos = np.zeros(2, dtype=np.float32)

        # --- AgentA: append token distance scalar ---
        pos_a = self.env.agents_pos["A"].astype(np.float32)
        token_dist_a = float(np.linalg.norm(pos_a - self._token_pos)) if self._token_active else 0.0
        obs["A"] = torch.cat([obs["A"], torch.tensor([token_dist_a], dtype=torch.float32)])

        # --- AgentB: stack token presence channel onto volumetric map ---
        token_channel = torch.zeros(
            1, self.env.z_layers, self.env.grid_size, self.env.grid_size
        )
        if self._token_active:
            tx, ty = int(self._token_pos[0]), int(self._token_pos[1])
            token_channel[0, :, tx, ty] = 1.0  # token present across all z-layers
        obs["B"] = torch.cat([obs["B"], token_channel], dim=0)

        # --- AgentC: append 12 distances from token to all existing entities ---
        # Entity order matches _get_relational_features: target, obs1..N, A, B, C
        entities = [self.env.target_pos.astype(np.float32)]
        entities.extend(o.astype(np.float32) for o in self.env.obstacles)
        for name in ["A", "B", "C"]:
            entities.append(self.env.agents_pos[name].astype(np.float32))
        token_features = [
            float(np.linalg.norm(token_pos - ent)) if self._token_active else 0.0
            for ent in entities
        ]
        obs["C"] = torch.cat(
            [obs["C"], torch.tensor(token_features, dtype=torch.float32)]
        )

        return obs

    def _run_epoch(self, epoch: int) -> dict:
        """Run one epoch (multiple episodes) and collect metrics."""
        epoch_rewards = {name: 0.0 for name in ["A", "B", "C"]}
        epoch_survivals = 0
        epoch_target_reached = 0
        epoch_steps = 0
        epoch_energy_spent = 0.0
        epoch_energy_delta_sum = {"A": 0.0, "B": 0.0, "C": 0.0}
        epoch_sacrifice_choices: list[int] = []   # accumulated across episodes
        epoch_self_state_norms: list[float] = []  # accumulated across episodes (depth=2 only)

        # Reset per-epoch protocol state (no-op for P0/P1; used by P2)
        self.protocol.reset_epoch()

        # Current temperature (Protocol 1/2: Gumbel-Softmax annealing; Protocol 0: unused)
        tau = self.protocol.get_tau(epoch)

        for agent in self.agents:
            agent.clear_episode()

        last_trajectory = None
        episode_summaries: list[dict] = []   # per-episode type breakdown for counter-wave
        pressure_hold_remaining = 0          # Exp 5: episodes remaining under elevated tax
        for episode in range(self.config.episodes_per_epoch):
            # Exp 5: elevated tax after a full-survival episode
            if self.config.counter_wave_mode == 5 and pressure_hold_remaining > 0:
                effective_tax = self.config.communication_tax_rate * self.config.post_success_tax_multiplier
            else:
                effective_tax = self.config.communication_tax_rate

            # Reset per-episode protocol state (no-op for P0/P1; used by P2)
            self.protocol.reset_episode()

            type_history_start = len(self.comm_buffer.type_history)
            record_trajectory = (episode == self.config.episodes_per_epoch - 1)
            result = self._run_episode(
                epoch,
                tau=tau,
                record_trajectory=record_trajectory,
                effective_tax_rate=effective_tax,
                no_boundary=(self.config.counter_wave_mode == 3),
            )

            # Per-episode type distribution for counter-wave analysis
            ep_types = self.comm_buffer.type_history[type_history_start:]
            counts = {0: 0, 1: 0, 2: 0}
            for step in ep_types:
                for t in step.values():
                    counts[t] = counts.get(t, 0) + 1
            ep_total = max(sum(counts.values()), 1)
            episode_summaries.append({
                "survived": result["survived"],
                "target_reached": result["target_reached"],
                "n_steps": result["steps"],
                "declare_rate": round(counts[0] / ep_total, 4),
                "query_rate":   round(counts[1] / ep_total, 4),
                "respond_rate": round(counts[2] / ep_total, 4),
                "success_step": result.get("success_step"),
                "pressure_held": (self.config.counter_wave_mode == 5 and pressure_hold_remaining > 0),
            })

            # Exp 5: trigger pressure hold after full-survival episode
            if self.config.counter_wave_mode == 5 and result.get("full_survival", False):
                pressure_hold_remaining = self.config.post_success_pressure_episodes
            elif pressure_hold_remaining > 0:
                pressure_hold_remaining -= 1

            for name in ["A", "B", "C"]:
                epoch_rewards[name] += result["total_rewards"][name]
                epoch_energy_delta_sum[name] += result.get("energy_delta_mean", {}).get(name, 0.0)
            epoch_survivals += 1 if result["survived"] else 0
            epoch_target_reached += 1 if result["target_reached"] else 0
            epoch_steps += result["steps"]
            epoch_energy_spent += result["energy_spent"]
            epoch_sacrifice_choices.extend(result.get("sacrifice_choices", []))
            epoch_self_state_norms.extend(result.get("self_state_norms", []))
            if record_trajectory:
                last_trajectory = result.get("trajectory")

        # Training update — skipped in phase 2 of hysteresis (all params frozen)
        if not self.config.training_frozen:
            losses = update_agents(self.agents, self.optimizers, self.config.gamma)
        else:
            losses = {a.name: 0.0 for a in self.agents}
            for agent in self.agents:
                agent.clear_episode()

        # Compute signal metrics from comm buffer history (BEFORE reset)
        history = self.comm_buffer.history
        type_history = self.comm_buffer.type_history
        entropy = compute_per_agent_entropy(history)
        mi = compute_pairwise_mi(history)
        te = compute_all_pairs_te(history)
        zipf = compute_zipf_per_agent(history)

        avg_episodes = self.config.episodes_per_epoch
        target_rate = epoch_target_reached / avg_episodes
        avg_energy = epoch_energy_spent / avg_episodes
        roi = compute_energy_roi(target_rate, avg_energy)

        # Protocol-specific extras (inquiry metrics for P1, empty dict for P0)
        extras = self.protocol.compute_epoch_extras(
            type_history=type_history,
            signal_history=history,
            target_rate=target_rate,
            tax_rate=self.config.communication_tax_rate,
        )

        # Collect PCA signal samples before buffer reset
        pca_samples = collect_signal_samples(history, type_history)
        self._all_signal_samples.append(pca_samples)

        # Clear comm buffer history for next epoch
        self.comm_buffer.reset()

        metrics = {
            "epoch": epoch,
            "avg_reward": {k: v / avg_episodes for k, v in epoch_rewards.items()},
            "energy_delta_mean": {name: epoch_energy_delta_sum[name] / avg_episodes for name in ["A", "B", "C"]},
            "survival_rate": epoch_survivals / avg_episodes,
            "target_reached_rate": target_rate,
            "avg_steps": epoch_steps / avg_episodes,
            "avg_energy_spent": avg_energy,
            "losses": losses,
            "entropy": entropy,
            "mutual_information": mi,
            "transfer_entropy": te,
            "zipf": zipf,
            "energy_roi": roi,
            "comm_killed": self.comm_buffer.is_killed,
            "timestamp": time.time(),
            "trajectory": last_trajectory,
            "tau": tau,
            "episode_summaries": episode_summaries,  # per-episode type breakdown
            # Protocol 4: Sacrifice-Conflict and depth tracking
            "sacrifice_choices": epoch_sacrifice_choices,
            "sacrifice_choice_rate": (
                float(sum(epoch_sacrifice_choices)) / len(epoch_sacrifice_choices)
                if epoch_sacrifice_choices else None
            ),
            "self_state_norm_mean": (
                float(np.mean(epoch_self_state_norms)) if epoch_self_state_norms else None
            ),
        }
        metrics.update(extras)  # merges 'inquiry' for P1/P2, 'ethical_constraint' for P2, nothing for P0

        # Protocol 2: compute collapse metrics using accumulated query-rate series
        if isinstance(self.protocol, Protocol2):
            query_rates = list(self.protocol._epoch_query_rates)  # updated by compute_epoch_extras
            collapse_result = interrogative_collapse_rate(query_rates)
            # H1 proxy: fraction of epochs where QUERY rate stays below collapse_threshold.
            # interrogative_collapse_rate is the correct operationalization of exploitation loops
            # in this framework -- an agent not generating queries is locked in non-interrogative
            # behavior. exploitation_loop_detection() remains in collapse_metrics.py for future
            # use with actual resource target IDs (requires environment-level tracking).
            loop_result = interrogative_collapse_rate(query_rates)
            metrics["collapse_metrics"] = {
                "interrogative_collapse": collapse_result,
                "exploitation_loop": loop_result,
            }

        return metrics

    def _run_episode(
        self,
        epoch_seed_offset: int,
        tau: float = 1.0,
        record_trajectory: bool = False,
        effective_tax_rate: float | None = None,  # Exp 5: elevated pressure override
        no_boundary: bool = False,                # Exp 3: continue for N steps after success
    ) -> dict:
        """Run a single episode and return summary."""
        episode_seed = self.config.seed + epoch_seed_offset
        obs = self.env.reset(seed=episode_seed)
        obs = self._augment_obs_with_token(obs)

        self.agent_a.reset_hidden()

        # Sacrifice-Conflict episode state reset
        self._token_pos = None
        self._token_active = False
        self._critical_agent = None
        self._token_trigger_step = None
        _sacrifice_start_pos: dict[str, np.ndarray] = {}   # non-critical agents at trigger
        _sacrifice_token_pos: np.ndarray | None = None      # token position at trigger
        _sacrifice_choices_evaluated: bool = False
        episode_sacrifice_choices: list[int] = []
        episode_self_state_norms: list[float] = []

        # Energy delta tracking — pre-build requirement for Protocol 4 Depth 2
        # Captures per-step energy consumption delta per agent.
        # self_model_gru (Depth 2) will read energy_delta as one of its 4 inputs.
        energy_prev: dict[str, float] = {name: self.env.energy[name] for name in ["A", "B", "C"]}
        episode_energy_deltas: dict[str, list[float]] = {name: [] for name in ["A", "B", "C"]}

        # Counter-wave experiment overrides
        tax_rate = effective_tax_rate if effective_tax_rate is not None else self.config.communication_tax_rate
        survival_bonus = 0.0 if self.config.counter_wave_mode == 4 else self.config.survival_bonus

        total_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
        done = False
        success_step: int | None = None    # Exp 3: step at which target was reached
        post_success_remaining: int = 0    # Exp 3: steps left after success

        trajectory = None
        if record_trajectory:
            trajectory = {
                "grid_size": self.env.grid_size,
                "target": self.env.target_pos.tolist(),
                "obstacles": [o.tolist() for o in self.env.obstacles],
                "steps": [],
            }

        while not done:
            signals = {}
            signal_types = {}
            action_logits = {}
            all_type_logits = {}

            for agent in self.agents:
                agent_obs = obs[agent.name].to(self.device)
                incoming = self.comm_buffer.receive_all(agent.name).to(self.device)

                signal, logits, type_logits = agent(agent_obs, incoming)

                # Depth 2: collect self_state L2 norm per step for epoch logging
                if agent.name == "A" and self.config.depth >= 2:
                    ss = self.agent_a._self_state
                    if ss is not None:
                        episode_self_state_norms.append(ss.detach().norm().item())

                # Protocol-aware type resolution:
                # P0: returns (None, 0) — no gradient, always DECLARE
                # P1: returns (soft_type, hard_type) — Gumbel-Softmax
                soft_type, hard_type = self.protocol.resolve_signal_type(
                    type_logits, tau, training=True
                )

                signals[agent.name] = signal
                signal_types[agent.name] = hard_type
                action_logits[agent.name] = logits
                all_type_logits[agent.name] = type_logits

            # Store signals and types in comm buffer
            for name, signal in signals.items():
                self.comm_buffer.send(name, signal, signal_type=signal_types[name])
            self.comm_buffer.record_history()

            # Sample actions and compute combined log probs
            actions = {}
            for agent in self.agents:
                logits = action_logits[agent.name]
                dist = Categorical(logits=logits)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)

                # Include type_head gradient only if protocol trains it AND this
                # agent's type_head is not frozen (supports per-agent ablation).
                type_head_active = (
                    self.protocol.should_train_type_head()
                    and any(p.requires_grad for p in agent.type_head.parameters())
                )
                if type_head_active:
                    type_dist = Categorical(logits=all_type_logits[agent.name])
                    type_log_prob = type_dist.log_prob(
                        torch.tensor(signal_types[agent.name], device=self.device)
                    )
                    combined_log_prob = action_log_prob + type_log_prob
                else:
                    combined_log_prob = action_log_prob

                agent.store_outcome(combined_log_prob, 0.0)  # reward filled below
                actions[agent.name] = action.item()

            # Environment step
            obs, env_rewards, done, info = self.env.step(actions)

            # Sacrifice-Conflict: claim check BEFORE obs augmentation.
            # Agents moved this step — check if any reached the token.
            if self._token_active and self._token_pos is not None:
                for _cn in ["A", "B", "C"]:
                    if np.array_equal(self.env.agents_pos[_cn], self._token_pos):
                        self.env.energy[_cn] = 100.0   # full restoration
                        info["energy"][_cn] = 100.0    # keep info consistent
                        self._token_active = False
                        break

            # Sacrifice-Conflict: trigger check.
            # Fire only once per episode (critical_agent tracks first trigger).
            if not self._token_active and self._critical_agent is None:
                _trigger_name: str | None = None
                _min_e = self.config.critical_energy_threshold
                for _tn in ["A", "B", "C"]:
                    if info["energy"][_tn] < self.config.critical_energy_threshold and info["energy"][_tn] < _min_e:
                        _trigger_name = _tn
                        _min_e = info["energy"][_tn]
                if _trigger_name is not None:
                    # Spawn token at random unoccupied cell
                    _occupied = {tuple(self.env.target_pos)}
                    _occupied |= {tuple(o) for o in self.env.obstacles}
                    _occupied |= {tuple(self.env.agents_pos[n]) for n in ["A", "B", "C"]}
                    while True:
                        _cand = np.array([
                            np.random.randint(0, self.env.grid_size),
                            np.random.randint(0, self.env.grid_size),
                        ])
                        if tuple(_cand) not in _occupied:
                            break
                    self._token_pos = _cand.astype(np.float32)
                    self._token_active = True
                    self._critical_agent = _trigger_name
                    self._token_trigger_step = info["step"]
                    # Record positions for sacrifice path-choice logging
                    _sacrifice_token_pos = self._token_pos.copy()
                    for _sn in ["A", "B", "C"]:
                        if _sn != _trigger_name:
                            _sacrifice_start_pos[_sn] = self.env.agents_pos[_sn].copy()

            obs = self._augment_obs_with_token(obs)

            # Sacrifice choice evaluation: classify at window end (trigger_step + N)
            if (
                not _sacrifice_choices_evaluated
                and self._token_trigger_step is not None
                and _sacrifice_token_pos is not None
                and info["step"] >= self._token_trigger_step + _SACRIFICE_WINDOW
            ):
                for _sn, _sp in _sacrifice_start_pos.items():
                    _d0 = float(np.linalg.norm(_sp.astype(np.float32) - _sacrifice_token_pos))
                    _d1 = float(np.linalg.norm(
                        self.env.agents_pos[_sn].astype(np.float32) - _sacrifice_token_pos
                    ))
                    episode_sacrifice_choices.append(1 if _d1 > _d0 else 0)
                _sacrifice_choices_evaluated = True

            # Compute per-step energy delta (negative = energy consumed this step)
            step_energy_delta = {
                name: info["energy"][name] - energy_prev[name]
                for name in ["A", "B", "C"]
            }
            for name in ["A", "B", "C"]:
                episode_energy_deltas[name].append(step_energy_delta[name])
            energy_prev = dict(info["energy"])
            info["energy_delta"] = step_energy_delta

            # Feed energy delta to AgentA for next step's self_model_gru (depth=2 only)
            if self.config.depth >= 2:
                self.agent_a.set_energy_delta(step_energy_delta["A"])

            # Exp 3 (no_boundary): on first target_reached, record success_step and
            # continue for post_success_steps more steps instead of terminating.
            if no_boundary and done and info.get("done_reason") == "target_reached" and success_step is None:
                success_step = info["step"]
                done = False
                post_success_remaining = self.config.post_success_steps
            elif no_boundary and success_step is not None and not done:
                if post_success_remaining > 0:
                    post_success_remaining -= 1
                    if post_success_remaining == 0:
                        done = True  # terminate after the extension window

            if trajectory is not None:
                trajectory["steps"].append({
                    "A": self.env.agents_pos["A"].tolist(),
                    "B": self.env.agents_pos["B"].tolist(),
                    "C": self.env.agents_pos["C"].tolist(),
                    "energy": {k: round(v, 1) for k, v in info["energy"].items()},
                    "energy_delta": {k: round(v, 3) for k, v in step_energy_delta.items()},
                })

            # Compute rewards via protocol (uses overridden tax_rate and survival_bonus)
            # Pass 1: individual protocol reward for each agent
            ind_rewards: dict[str, float] = {}
            for agent in self.agents:
                ind_rewards[agent.name] = self.protocol.compute_reward(
                    agent_name=agent.name,
                    env_reward=env_rewards[agent.name],
                    signal_sent=signals[agent.name],
                    energy_remaining=info["energy"][agent.name],
                    energy_budget=self.config.energy_budget,
                    communication_tax_rate=tax_rate,
                    reached_target=info["reached_target"][agent.name],
                    survival_bonus=survival_bonus,
                    signal_type=signal_types[agent.name],
                )
            # Pass 2: apply welfare coupling if enabled (Protocol 5)
            # R_i = 0.5 * self_reward + 0.5 * mean(other_agent_rewards), alpha=0.5
            for agent in self.agents:
                if self.config.welfare_coupled:
                    others_mean = sum(
                        v for k, v in ind_rewards.items() if k != agent.name
                    ) / (len(ind_rewards) - 1)
                    reward = 0.5 * ind_rewards[agent.name] + 0.5 * others_mean
                else:
                    reward = ind_rewards[agent.name]
                agent.rewards[-1] = reward
                total_rewards[agent.name] += reward

        # Evaluate sacrifice choices if episode ended before the window completed
        if not _sacrifice_choices_evaluated and _sacrifice_token_pos is not None:
            for _sn, _sp in _sacrifice_start_pos.items():
                _d0 = float(np.linalg.norm(_sp.astype(np.float32) - _sacrifice_token_pos))
                _d1 = float(np.linalg.norm(
                    self.env.agents_pos[_sn].astype(np.float32) - _sacrifice_token_pos
                ))
                episode_sacrifice_choices.append(1 if _d1 > _d0 else 0)

        target_reached = (info["done_reason"] == "target_reached") or (success_step is not None)
        result = {
            "total_rewards": total_rewards,
            "survived": info["done_reason"] != "energy_depleted",
            "target_reached": target_reached,
            "full_survival": target_reached,  # alias used by Exp 5 pressure-hold trigger
            "steps": info["step"],
            "energy_spent": self.env.total_energy_spent,
            "success_step": success_step,     # Exp 3: step at which success was detected (or None)
            # Per-agent mean energy delta over this episode (negative = net consumption)
            "energy_delta_mean": {
                name: (sum(deltas) / len(deltas)) if deltas else 0.0
                for name, deltas in episode_energy_deltas.items()
            },
            "sacrifice_choices": episode_sacrifice_choices,
            "self_state_norms": episode_self_state_norms,
        }
        if trajectory is not None:
            result["trajectory"] = trajectory
        return result

    # --- Manifest helpers ---

    def _write_manifest(self, path: str) -> None:
        """Write a run summary manifest to disk."""
        import os
        if not self.epoch_metrics:
            return
        os.makedirs(self.config.output_dir, exist_ok=True)
        full_path = os.path.join(self.config.output_dir, path)
        manifest = {
            "protocol": self.config.protocol,
            "seed": self.config.seed,
            "population_mode": self.config.population_mode if self.config.protocol == 2 else None,
            "declare_cost": self.config.declare_cost,
            "query_cost": self.config.query_cost,
            "respond_cost": self.config.respond_cost,
            "depth": self.config.depth,
            "ablation_active": self.config.ablate_self_model_inputs,
            "freeze_self_model_gru": self.config.freeze_self_model_gru,
            "welfare_coupled": self.config.welfare_coupled,
            "critical_energy_threshold": self.config.critical_energy_threshold,
            "epochs_total": len(self.epoch_metrics),
            "final_metrics": self._extract_final_metrics(self.epoch_metrics[-1]),
            "crystallization_epoch": self._find_crystallization_epoch(self.epoch_metrics),
            "per_agent_crystallization": self._find_per_agent_crystallization(self.epoch_metrics),
            "phase_transitions": self._detect_phase_transitions(self.epoch_metrics),
            "performance_stats": self._compute_performance_stats(self.epoch_metrics),
            "counter_wave_data": self._extract_counter_wave_data(self.epoch_metrics),
        }
        with open(full_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _write_epoch_series(self, path: str) -> None:
        """Write per-epoch metrics needed for confirmatory analysis.

        Backward-compatible: all fields from Protocol 2 are preserved.
        New Protocol 4 fields added alongside existing fields.

        Fields per epoch:
            epoch, type_entropy, qrc, query_rate, ethical_cost, collapse_detected,
            energy_delta_mean   — unchanged from Protocol 2
            depth               — depth level for this run (0/1/2)
            ablation_active     — whether self_model_gru inputs were ablated
            sacrifice_choices   — list of per-episode sacrifice choices (epoch accumulated)
            sacrifice_choice_rate -- mean sacrifice choice rate over epoch
            framework_scores    — per-agent four-framework ethical scores
            deception_metric    — per-agent std across four framework scores
            convergence_divergence_index -- Pearson(sacrifice_rate, mean_framework) over 50-epoch window
            self_state_norm     -- mean L2 norm of AgentA self_state (Depth 2 only; null at 0/1)
        """
        import os
        if not self.epoch_metrics:
            return
        os.makedirs(self.config.output_dir, exist_ok=True)
        full_path = os.path.join(self.config.output_dir, path)

        # Build base records with existing + new runtime fields
        series = []
        for m in self.epoch_metrics:
            inq = m.get("inquiry", {}) or {}
            ec = m.get("ethical_constraint", {}) or {}
            coll = m.get("collapse_metrics", {}).get("interrogative_collapse", {}) or {}
            ethical_cost = sum(ec.get("ethical_cost_by_agent", {}).values())

            framework_scores = self._compute_framework_scores(m)
            deception_metric = self._compute_deception_metric(framework_scores)

            series.append({
                # --- Existing Protocol 2 fields (unchanged) ---
                "epoch": m["epoch"],
                "type_entropy": inq.get("type_entropy"),
                "qrc": inq.get("query_response_coupling"),
                "query_rate": (inq.get("type_distribution") or {}).get("QUERY"),
                "ethical_cost": round(ethical_cost, 4),
                "collapse_detected": coll.get("collapse_detected", False),
                "energy_delta_mean": m.get("energy_delta_mean"),
                # --- Protocol 4 fields ---
                "depth": self.config.depth,
                "ablation_active": self.config.ablate_self_model_inputs,
                "freeze_self_model_gru": self.config.freeze_self_model_gru,
                "sacrifice_choices": m.get("sacrifice_choices", []),
                "sacrifice_choice_rate": m.get("sacrifice_choice_rate"),
                "framework_scores": framework_scores,
                "deception_metric": deception_metric,
                "convergence_divergence_index": None,  # filled in post-loop below
                "self_state_norm": m.get("self_state_norm_mean"),
            })

        # Convergence/Divergence Index: Pearson(sacrifice_choice_rate, mean_framework_score)
        # over trailing 50-epoch window.  Requires >= 50 valid epochs to produce a value.
        WINDOW = 50
        for i, rec in enumerate(series):
            window_start = max(0, i + 1 - WINDOW)
            window = series[window_start: i + 1]
            xs, ys = [], []
            for w in window:
                scr = w.get("sacrifice_choice_rate")
                fs = w.get("framework_scores") or {}
                # Mean framework score across all agents and all four frameworks
                vals = []
                for agent_scores in fs.values():
                    if isinstance(agent_scores, dict):
                        vals.extend(v for v in agent_scores.values() if v is not None)
                mfs = sum(vals) / len(vals) if vals else None
                if scr is not None and mfs is not None:
                    xs.append(scr)
                    ys.append(mfs)
            if len(xs) >= WINDOW:
                rec["convergence_divergence_index"] = self._pearsonr(xs, ys)
            # else: stays None

        with open(full_path, "w") as f:
            json.dump(series, f, indent=2)

    def _compute_framework_scores(self, m: dict) -> dict:
        """Compute four ethical framework scores per agent from epoch metrics.

        Scoring functions are post-hoc; no new data is gathered during the run.
        - Utilitarian: avg_steps / max_steps — normalized mean survival steps (global)
        - Deontological: 1 - exploitation_loop_rate (from collapse_metrics if available)
        - Virtue ethics: per-agent QUERY rate over the epoch
        - Self-interest: per-agent avg_reward normalized by MAX_REWARD heuristic (15.0)
        """
        MAX_REWARD = 15.0   # heuristic upper bound for per-episode total reward
        utilitarian = round(m.get("avg_steps", 0.0) / self.config.max_steps, 4)

        # Deontological: 1 - exploitation_loop_rate.
        # Protocol 2 collapse_metrics expose area_under_query_curve (trapezoidal AUC of
        # per-epoch QUERY rate). Normalized by epoch count it gives mean query compliance —
        # the continuous proxy for "not exploiting". For epoch 0 (single data point, AUC=0
        # from trapezoidal rule) fall back to the per-epoch inquiry query_rate directly.
        # For non-P2 runs where collapse_metrics is absent, also use inquiry query_rate.
        inq = m.get("inquiry", {}) or {}
        cm = m.get("collapse_metrics", {}) or {}
        el = cm.get("exploitation_loop", {}) or {}
        auc = el.get("area_under_query_curve", None)
        n_epochs = m.get("epoch", 0) + 1   # epoch is 0-indexed; +1 = epochs elapsed
        if auc is not None and n_epochs > 1:
            # mean query-compliance rate over all epochs so far (trapezoidal AUC / intervals)
            deontological = round(min(float(auc) / (n_epochs - 1), 1.0), 4)
        else:
            # epoch 0 fallback or non-P2: use this epoch's QUERY rate from inquiry
            epoch_qr = float((inq.get("type_distribution") or {}).get("QUERY") or 0.0)
            deontological = round(epoch_qr, 4) if epoch_qr > 0 else None
        per_agent_types = inq.get("per_agent_types", {}) or {}
        avg_reward = m.get("avg_reward", {}) or {}

        scores: dict[str, dict] = {}
        for name, key in [("A", "agent_a"), ("B", "agent_b"), ("C", "agent_c")]:
            query_rate = float(
                (per_agent_types.get(name, {}) or {}).get("QUERY", 0.0) or 0.0
            )
            raw_reward = avg_reward.get(name, 0.0) or 0.0
            self_interest = round(min(max(raw_reward / MAX_REWARD, 0.0), 1.0), 4)
            scores[key] = {
                "utilitarian": utilitarian,
                "deontological": deontological,
                "virtue_ethics": round(query_rate, 4),
                "self_interest": self_interest,
            }
        return scores

    def _compute_deception_metric(self, framework_scores: dict) -> dict:
        """Per-agent std across the four framework scores (equal-weighted).

        A high value indicates inconsistency across ethical frameworks —
        the agent scores well on some frameworks but poorly on others,
        suggesting strategic rather than principled behavior.
        """
        dm: dict[str, float | None] = {}
        for key, scores in framework_scores.items():
            vals = [v for v in scores.values() if v is not None]
            if len(vals) >= 2:
                mean = sum(vals) / len(vals)
                variance = sum((v - mean) ** 2 for v in vals) / len(vals)
                dm[key] = round(variance ** 0.5, 4)
            else:
                dm[key] = None
        return dm

    @staticmethod
    def _pearsonr(x: list[float], y: list[float]) -> float | None:
        """Pearson correlation coefficient between x and y."""
        n = len(x)
        if n < 2:
            return None
        mx = sum(x) / n
        my = sum(y) / n
        sx = sum((xi - mx) ** 2 for xi in x) ** 0.5
        sy = sum((yi - my) ** 2 for yi in y) ** 0.5
        if sx == 0.0 or sy == 0.0:
            return None
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
        return round(cov / (sx * sy), 4)

    def _extract_final_metrics(self, last_epoch: dict) -> dict:
        inq = last_epoch.get('inquiry', {})
        if not isinstance(inq, dict):
            inq = {}
        return {
            'survival_rate': last_epoch.get('survival_rate', 0),
            'target_reached_rate': last_epoch.get('target_reached_rate', 0),
            'energy_roi': last_epoch.get('energy_roi', 0),
            'type_entropy': inq.get('type_entropy', None),
            'qrc': inq.get('query_response_coupling', None),
            'per_agent_types': inq.get('per_agent_types', None),  # P4: per-arch participation
        }

    def _find_crystallization_epoch(self, epochs: list[dict]) -> int | None:
        """First epoch in a streak of 5 consecutive epochs with type_entropy < 0.95."""
        if self.config.protocol == 0:
            return None
        streak_start = None
        streak_count = 0
        for m in epochs:
            inq = m.get('inquiry', {})
            if not isinstance(inq, dict):
                streak_count = 0
                streak_start = None
                continue
            te = inq.get('type_entropy', None)
            if te is not None and te < 0.95:
                if streak_count == 0:
                    streak_start = m['epoch']
                streak_count += 1
                if streak_count >= 5:
                    return streak_start
            else:
                streak_count = 0
                streak_start = None
        return None

    def _find_per_agent_crystallization(self, epochs: list[dict]) -> dict[str, int | None]:
        """Per-agent crystallisation epoch: first epoch in a 5-streak where that
        agent's individual type entropy drops below 1.2 bits.

        H_max (uniform, 3 types) = log2(3) ≈ 1.585 bits.
        Threshold 1.2 bits requires one signal type to exceed ~60% share —
        meaningfully non-uniform and well above random initialisation.

        Used for preregistration P4 (substrate independence) ANOVA.
        """
        import math

        if self.config.protocol == 0:
            return {"A": None, "B": None, "C": None}

        THRESHOLD = 1.2  # bits

        results: dict[str, int | None] = {"A": None, "B": None, "C": None}
        streaks: dict[str, int] = {"A": 0, "B": 0, "C": 0}
        starts:  dict[str, int | None] = {"A": None, "B": None, "C": None}

        for m in epochs:
            inq = m.get("inquiry", {})
            if not isinstance(inq, dict):
                continue
            pat = inq.get("per_agent_types", {})
            for agent in ("A", "B", "C"):
                if results[agent] is not None:
                    continue
                ad = pat.get(agent) or {}
                d = ad.get("DECLARE", 0.0)
                q = ad.get("QUERY",   0.0)
                r = ad.get("RESPOND", 0.0)
                h = sum(-p * math.log2(p) for p in (d, q, r) if p > 1e-9)
                if h < THRESHOLD:
                    if streaks[agent] == 0:
                        starts[agent] = m["epoch"]
                    streaks[agent] += 1
                    if streaks[agent] >= 5:
                        results[agent] = starts[agent]
                else:
                    streaks[agent] = 0
                    starts[agent]  = None

        return results

    def _detect_phase_transitions(self, epochs: list[dict]) -> list[dict]:
        """Epochs where type_entropy drops by more than 0.05 in a single step."""
        transitions = []
        prev_te = None
        for m in epochs:
            inq = m.get('inquiry', {})
            if not isinstance(inq, dict):
                continue
            te = inq.get('type_entropy', None)
            if te is not None and prev_te is not None:
                delta = prev_te - te
                if delta > 0.05:
                    transitions.append({
                        'epoch': m['epoch'],
                        'entropy_before': round(prev_te, 4),
                        'entropy_after': round(te, 4),
                        'delta': round(delta, 4),
                    })
            prev_te = te
        return transitions

    def _compute_performance_stats(self, epochs: list[dict]) -> dict:
        survival_rates = [m.get('survival_rate', 0) for m in epochs]
        target_rates = [m.get('target_reached_rate', 0) for m in epochs]
        n = max(len(survival_rates), 1)
        return {
            'avg_survival_rate': round(sum(survival_rates) / n, 4),
            'avg_target_rate': round(sum(target_rates) / n, 4),
            'max_survival_rate': round(max(survival_rates) if survival_rates else 0, 4),
            'max_target_rate': round(max(target_rates) if target_rates else 0, 4),
        }

    def _extract_counter_wave_data(self, epochs: list[dict]) -> dict:
        """Extract counter-wave events and per-episode type breakdowns.

        A counter-wave event is defined as: an epoch with survival_rate=1.0 followed
        by an epoch where type_entropy (H) increases by >= 0.03 bits.

        Returns a dict with:
          - counter_wave_mode: experiment variant
          - events: list of detected counter-wave events with context window
          - full_survival_epochs: all epochs with survival_rate==1.0
          - declare_rate_at_success: mean DECLARE rate in full-survival episodes
          - declare_rate_baseline: mean DECLARE rate in non-success episodes
        """
        if self.config.protocol == 0:
            return {}

        full_survival_epochs = []
        declare_at_success: list[float] = []
        declare_baseline: list[float] = []

        for m in epochs:
            ep_sums = m.get('episode_summaries', [])
            for ep in ep_sums:
                if ep.get('target_reached'):
                    declare_at_success.append(ep.get('declare_rate', 0.0))
                else:
                    declare_baseline.append(ep.get('declare_rate', 0.0))
            if m.get('survival_rate', 0) >= 1.0:
                full_survival_epochs.append(m['epoch'])

        # Counter-wave events: full-survival epoch followed by H increase
        events = []
        for i, m in enumerate(epochs[:-1]):
            if m.get('survival_rate', 0) < 1.0:
                continue
            h_now  = (m.get('inquiry') or {}).get('type_entropy')
            h_next = (epochs[i + 1].get('inquiry') or {}).get('type_entropy')
            if h_now is None or h_next is None:
                continue
            delta_h = h_next - h_now
            if delta_h >= 0.03:
                # Build K=2 episode context window
                ep_before = m.get('episode_summaries', [])
                ep_after  = epochs[i + 1].get('episode_summaries', [])
                events.append({
                    'epoch':       m['epoch'],
                    'h_before':    round(h_now, 4),
                    'h_after':     round(h_next, 4),
                    'delta_h':     round(delta_h, 4),
                    'success_episodes': [e for e in ep_before if e.get('target_reached')],
                    'post_success_episodes': ep_after[:3],  # first 3 episodes of next epoch
                })

        import statistics as st
        return {
            'counter_wave_mode': self.config.counter_wave_mode,
            'n_full_survival_epochs': len(full_survival_epochs),
            'n_counter_wave_events': len(events),
            'declare_rate_at_success': round(st.mean(declare_at_success), 4) if declare_at_success else None,
            'declare_rate_baseline':  round(st.mean(declare_baseline), 4)  if declare_baseline  else None,
            'full_survival_epochs': full_survival_epochs[:50],  # cap at 50 for JSON size
            'events': events[:20],                              # cap at 20 events
        }

    # --- Control methods ---

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._stopped = True
        self._paused = False

    def kill_communication(self) -> None:
        self.comm_buffer.kill()

    def restore_communication(self) -> None:
        self.comm_buffer.restore()

    @property
    def is_running(self) -> bool:
        return not self._stopped and not self._paused

    def get_agent_weights(self) -> dict[str, dict]:
        """Export all agent state dicts."""
        return {agent.name: agent.state_dict() for agent in self.agents}

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """Save full simulation state for checkpoint/resume across epoch boundaries."""
        torch.save({
            "epoch": epoch,
            "agent_states":     {a.name: a.state_dict() for a in self.agents},
            "optimizer_states": [o.state_dict() for o in self.optimizers],
            "epoch_metrics":    self.epoch_metrics,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return the epoch to resume from."""
        ckpt = torch.load(path, weights_only=False)
        for agent in self.agents:
            agent.load_state_dict(ckpt["agent_states"][agent.name])
        for opt, state in zip(self.optimizers, ckpt["optimizer_states"]):
            opt.load_state_dict(state)
        self.epoch_metrics = ckpt["epoch_metrics"]
        return int(ckpt["epoch"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MARL simulation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--protocol", type=int, default=1, choices=[0, 1, 2],
                        help="0=Baseline, 1=Interrogative Emergence, 2=Ethical Constraints")
    parser.add_argument("--population-mode", type=str, default="all_unconstrained",
                        choices=["all_unconstrained", "all_constrained"],
                        help="Protocol 2 condition (default: all_unconstrained)")
    parser.add_argument("--declare-cost", type=float, default=1.0,
                        help="DECLARE signal cost multiplier (Protocol 1)")
    parser.add_argument("--query-cost", type=float, default=1.5,
                        help="QUERY signal cost multiplier (Protocol 1)")
    parser.add_argument("--respond-cost", type=float, default=0.8,
                        help="RESPOND signal cost multiplier (Protocol 1)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for manifest and report output")
    parser.add_argument("--ablate-agent-c", action="store_true",
                        help="Experiment 6: freeze Agent C type_head only")
    parser.add_argument("--counter-wave-mode", type=int, default=0, choices=[0, 3, 4, 5],
                        help="Counter-wave experiment: 0=normal, 3=no-boundary, 4=no-bonus, 5=pressure-hold")
    args = parser.parse_args()

    def print_metrics(m: dict) -> None:
        inq = m.get('inquiry', {})
        te_str = (
            f" | H: {inq['type_entropy']:.3f}"
            if isinstance(inq, dict) and 'type_entropy' in inq
            else ""
        )
        print(
            f"Epoch {m['epoch']:3d} | "
            f"Survival: {m['survival_rate']:.2f} | "
            f"Target: {m['target_reached_rate']:.2f} | "
            f"Reward A: {m['avg_reward']['A']:.3f}"
            f"{te_str}"
        )

    config = SimulationConfig(
        seed=args.seed,
        num_epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        protocol=args.protocol,
        declare_cost=args.declare_cost,
        query_cost=args.query_cost,
        respond_cost=args.respond_cost,
        population_mode=args.population_mode,
        output_dir=args.output_dir,
        ablate_agent_c_type_head=args.ablate_agent_c,
        counter_wave_mode=args.counter_wave_mode,
    )
    engine = SimulationEngine(config=config, epoch_callback=print_metrics)
    engine.run()
    print("Simulation complete.")

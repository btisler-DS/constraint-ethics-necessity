"""Protocol registry for the MARL simulation.

Protocols define the reward structure, signal type semantics, and metrics
computed per epoch. Switching protocols does NOT change the agent architecture —
only the learning dynamics and reward signals.

Protocol 0: Baseline — flat Landauer tax, no type head gradient.
Protocol 1: Interrogative Emergence — Gumbel-Softmax type head, differential costs.
Protocol 2: Ethical Constraints — ethical tax as Landauer-style resource cost on
            exploitation loops. Two conditions: all_constrained / all_unconstrained.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .training.reward import compute_reward
from .training.temperature import get_tau, sample_signal_type
from .metrics.inquiry_metrics import compute_inquiry_metrics


class ProtocolBase(ABC):
    """Abstract protocol interface."""

    protocol_id: int

    @abstractmethod
    def get_tau(self, epoch: int) -> float:
        """Return current Gumbel-Softmax temperature."""

    @abstractmethod
    def resolve_signal_type(
        self,
        type_logits: Tensor,
        tau: float,
        training: bool = True,
    ) -> tuple[Tensor | None, int]:
        """Determine signal type from logits.

        Returns:
            (soft_type | None, hard_type_int)
            soft_type is None for Protocol 0 (no gradient through type head).
        """

    @abstractmethod
    def compute_reward(
        self,
        *,
        agent_name: str,
        env_reward: float,
        signal_sent: Tensor,
        energy_remaining: float,
        energy_budget: float,
        communication_tax_rate: float,
        reached_target: bool,
        survival_bonus: float,
        signal_type: int,
    ) -> float:
        """Compute the reward for a single agent step."""

    @abstractmethod
    def compute_epoch_extras(
        self,
        *,
        type_history: list,
        signal_history: list,
        target_rate: float,
        tax_rate: float,
    ) -> dict:
        """Compute protocol-specific metrics for the epoch."""

    def should_train_type_head(self) -> bool:
        """Whether the type head should receive REINFORCE gradients."""
        return True

    def reset_episode(self) -> None:
        """Called at the start of each episode. Override in stateful protocols."""

    def reset_epoch(self) -> None:
        """Called at the start of each epoch. Override in stateful protocols."""


class Protocol0(ProtocolBase):
    """Baseline protocol — flat Landauer tax, no type head gradient.

    Reproduces exact Run 10 behaviour: signal_type is always 0 (DECLARE),
    the type head is frozen, and no inquiry metrics are logged.
    """

    protocol_id = 0

    def get_tau(self, epoch: int) -> float:
        return 1.0  # unused — type head frozen

    def resolve_signal_type(
        self, type_logits: Tensor, tau: float, training: bool = True
    ) -> tuple[None, int]:
        return None, 0  # Always DECLARE, no gradient

    def compute_reward(
        self,
        *,
        agent_name: str,
        env_reward: float,
        signal_sent: Tensor,
        energy_remaining: float,
        energy_budget: float,
        communication_tax_rate: float,
        reached_target: bool,
        survival_bonus: float,
        signal_type: int,
    ) -> float:
        # Flat tax — no type multiplier, identical to Run 10
        signal_cost = communication_tax_rate * signal_sent.abs().sum().item()
        energy_fraction = max(energy_remaining / energy_budget, 0.0)
        return env_reward - signal_cost + survival_bonus * energy_fraction

    def compute_epoch_extras(self, **_) -> dict:
        return {}  # No inquiry metrics for Protocol 0

    def should_train_type_head(self) -> bool:
        return False


class Protocol1(ProtocolBase):
    """Interrogative Emergence — Gumbel-Softmax, differential costs, inquiry metrics.

    Agents learn to distinguish DECLARATIVE / INTERROGATIVE / RESPONSE signals
    via a 3-way type head trained jointly with REINFORCE.

    Cost multipliers are configurable to support the five preregistered conditions:
        Baseline:      declare=1.0, query=1.5, respond=0.8
        Low Pressure:  declare=1.0, query=1.2, respond=0.9
        High Pressure: declare=1.0, query=3.0, respond=0.5
        Extreme:       declare=1.0, query=5.0, respond=0.3
    """

    protocol_id = 1

    def __init__(
        self,
        declare_cost: float = 1.0,
        query_cost: float = 1.5,
        respond_cost: float = 0.8,
    ) -> None:
        self.declare_cost = declare_cost
        self.query_cost = query_cost
        self.respond_cost = respond_cost

    def get_tau(self, epoch: int) -> float:
        return get_tau(epoch)

    def resolve_signal_type(
        self, type_logits: Tensor, tau: float, training: bool = True
    ) -> tuple[Tensor, int]:
        return sample_signal_type(type_logits, tau, training)

    def compute_reward(
        self,
        *,
        agent_name: str,
        env_reward: float,
        signal_sent: Tensor,
        energy_remaining: float,
        energy_budget: float,
        communication_tax_rate: float,
        reached_target: bool,
        survival_bonus: float,
        signal_type: int,
    ) -> float:
        return compute_reward(
            agent_name=agent_name,
            env_reward=env_reward,
            signal_sent=signal_sent,
            energy_remaining=energy_remaining,
            energy_budget=energy_budget,
            communication_tax_rate=communication_tax_rate,
            reached_target=reached_target,
            survival_bonus=survival_bonus,
            signal_type=signal_type,
            declare_cost=self.declare_cost,
            query_cost=self.query_cost,
            respond_cost=self.respond_cost,
        )

    def compute_epoch_extras(
        self,
        *,
        type_history: list,
        signal_history: list,
        target_rate: float,
        tax_rate: float,
    ) -> dict:
        return {
            'inquiry': compute_inquiry_metrics(
                type_history=type_history,
                signal_history=signal_history,
                target_reached_rate=target_rate,
                communication_tax_rate=tax_rate,
            )
        }


class Protocol2(ProtocolBase):
    """Ethical Constraints — ethical tax as Landauer-style resource cost on exploitation.

    Tests the architectural necessity of ethical constraints for sustaining interrogative
    complexity. Two conditions:

        all_unconstrained — no ethical tax (control / baseline)
        all_constrained   — ethical tax on all agents when in exploitation mode

    Exploitation is defined as consecutive steps without a QUERY signal:
        Step 1 = initial targeting (DECLARE: "I see the resource")
        Step 2 = confirmation (DECLARE: no new information acquired)
        Step 3 = lock-in (exploitation begins — Omega→Delta→Phi transition)

    FIXED constants (theoretically derived, locked for Preregistration v3):
        EXPLOIT_THRESHOLD = 3  — Omega→Delta→Phi coordination cycle
        ETHICAL_TAX_RATE  = 2.0 — Landauer multiplier on signal_cost during exploitation
    """

    protocol_id = 2

    EXPLOIT_THRESHOLD: int = 3    # Fixed — do not expose as configurable
    ETHICAL_TAX_RATE: float = 2.0  # Fixed — do not expose as configurable

    _VALID_MODES = frozenset({"all_unconstrained", "all_constrained"})

    def __init__(self, population_mode: str = "all_unconstrained") -> None:
        if population_mode not in self._VALID_MODES:
            raise ValueError(
                f"Invalid population_mode '{population_mode}'. "
                f"Valid values: {sorted(self._VALID_MODES)}"
            )
        self.population_mode = population_mode

        # Per-agent exploitation state (reset per episode)
        self._consecutive_nonquery: dict[str, int] = {"A": 0, "B": 0, "C": 0}
        self._exploitation_flags: dict[str, bool] = {"A": False, "B": False, "C": False}
        self._agent_step: dict[str, int] = {"A": 0, "B": 0, "C": 0}

        # Per-epoch accumulators (reset per epoch)
        self._epoch_ethical_cost: dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}
        self._exploitation_events: list[dict] = []

        # Epoch-level query-rate series (for collapse metrics in engine)
        self._epoch_query_rates: list[float] = []

    # --- ProtocolBase interface ---

    def get_tau(self, epoch: int) -> float:
        return get_tau(epoch)

    def resolve_signal_type(
        self, type_logits: Tensor, tau: float, training: bool = True
    ) -> tuple[Tensor, int]:
        return sample_signal_type(type_logits, tau, training)

    def compute_reward(
        self,
        *,
        agent_name: str,
        env_reward: float,
        signal_sent: Tensor,
        energy_remaining: float,
        energy_budget: float,
        communication_tax_rate: float,
        reached_target: bool,
        survival_bonus: float,
        signal_type: int,
    ) -> float:
        step = self._agent_step[agent_name]

        # Track exploitation: consecutive steps without QUERY (signal_type != 1)
        if signal_type == 1:  # QUERY breaks the exploitation loop
            self._consecutive_nonquery[agent_name] = 0
        else:
            self._consecutive_nonquery[agent_name] += 1

        was_exploiting = self._exploitation_flags[agent_name]
        is_exploiting = self._consecutive_nonquery[agent_name] >= self.EXPLOIT_THRESHOLD
        self._exploitation_flags[agent_name] = is_exploiting
        self._agent_step[agent_name] += 1

        if is_exploiting and not was_exploiting:
            self._exploitation_events.append({
                "agent": agent_name, "step": step, "event": "loop_entry",
                "consecutive_nonquery": self._consecutive_nonquery[agent_name],
            })
        elif not is_exploiting and was_exploiting:
            self._exploitation_events.append({
                "agent": agent_name, "step": step, "event": "loop_exit",
            })

        # Base signal cost (no type multiplier — P2 uses flat cost + ethical tax)
        signal_cost = communication_tax_rate * signal_sent.abs().sum().item()
        energy_fraction = max(energy_remaining / energy_budget, 0.0)

        # Ethical cost: applied only in all_constrained condition
        if self.population_mode == "all_constrained" and is_exploiting:
            ethical_cost = self.ETHICAL_TAX_RATE * signal_cost
        else:
            ethical_cost = 0.0

        self._epoch_ethical_cost[agent_name] += ethical_cost

        return env_reward - signal_cost - ethical_cost + survival_bonus * energy_fraction

    def compute_epoch_extras(
        self,
        *,
        type_history: list,
        signal_history: list,
        target_rate: float,
        tax_rate: float,
    ) -> dict:
        inquiry = compute_inquiry_metrics(
            type_history=type_history,
            signal_history=signal_history,
            target_reached_rate=target_rate,
            communication_tax_rate=tax_rate,
        )
        # Record query rate for collapse tracking (used by engine)
        td = inquiry.get("type_distribution") or {}
        self._epoch_query_rates.append(td.get("QUERY", 0.0))

        return {
            "inquiry": inquiry,
            "ethical_constraint": {
                "population_mode": self.population_mode,
                "ethical_cost_by_agent": {k: round(v, 6) for k, v in self._epoch_ethical_cost.items()},
                "exploitation_events": list(self._exploitation_events),
                "exploitation_flags_end_of_epoch": dict(self._exploitation_flags),
            },
        }

    # --- Lifecycle hooks ---

    def reset_episode(self) -> None:
        """Reset per-episode exploitation counters."""
        for agent in ("A", "B", "C"):
            self._consecutive_nonquery[agent] = 0
            self._exploitation_flags[agent] = False
            self._agent_step[agent] = 0

    def reset_epoch(self) -> None:
        """Reset per-epoch cost accumulators."""
        self._epoch_ethical_cost = {"A": 0.0, "B": 0.0, "C": 0.0}
        self._exploitation_events = []


PROTOCOL_REGISTRY: dict[int, type[ProtocolBase]] = {
    0: Protocol0,
    1: Protocol1,
    2: Protocol2,
}


def create_protocol(
    protocol_id: int,
    declare_cost: float = 1.0,
    query_cost: float = 1.5,
    respond_cost: float = 0.8,
    population_mode: str = "all_unconstrained",
) -> ProtocolBase:
    """Instantiate a protocol by ID.

    Args:
        protocol_id:     0=Baseline, 1=Interrogative Emergence, 2=Ethical Constraints.
        declare_cost:    DECLARE signal cost multiplier (Protocol 1 only).
        query_cost:      QUERY signal cost multiplier (Protocol 1 only).
        respond_cost:    RESPOND signal cost multiplier (Protocol 1 only).
        population_mode: Condition for Protocol 2 ('all_unconstrained'|'all_constrained').

    Raises:
        ValueError: If protocol_id is not in PROTOCOL_REGISTRY.
    """
    if protocol_id not in PROTOCOL_REGISTRY:
        raise ValueError(
            f"Unknown protocol: {protocol_id}. "
            f"Valid IDs: {sorted(PROTOCOL_REGISTRY)}"
        )
    if protocol_id == 1:
        return Protocol1(
            declare_cost=declare_cost,
            query_cost=query_cost,
            respond_cost=respond_cost,
        )
    if protocol_id == 2:
        return Protocol2(population_mode=population_mode)
    return PROTOCOL_REGISTRY[protocol_id]()

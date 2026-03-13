# Protocol 4 — Parameter Count Manifest

**Status:** Updated post obs-extension (token distance / presence added to all agents).
This document supersedes the parameter counts in the Protocol 4 briefing document.

---

## Observation Vector Dimensions (post extension)

| Agent | Before | After | Change |
|-------|--------|-------|--------|
| AgentA (1D sequence) | 11-dim | **12-dim** | +1 token distance scalar |
| AgentB (volumetric) | (1, Z, H, W) | **(2, Z, H, W)** | +1 token presence channel |
| AgentC (pairwise relational) | 66-dim | **78-dim** | +12 pairwise token distances |

AgentC entities (post extension): target + 8 obstacles + 3 agents + **token** = 13 entities
→ 13 × 12 / 2 = 78 pairwise distances.
The 12 new features: (token↔target), (token↔obs1..8), (token↔A), (token↔B), (token↔C).
When token is inactive all appended features are 0.0.

---

## Depth 0 — Feedforward Baseline (agent-appropriate input geometry)

Depth 0 strips each agent to a minimal feedforward trunk.
No GRU, no signal encoder, no cross-modal attention, no fusion.

### AgentA Depth 0 (obs_dim=12)

| Component | Shape | Params |
|-----------|-------|--------|
| obs_encoder | Linear(12, 64) | 768 + 64 = 832 |
| signal_mu | Linear(64, 8) | 512 + 8 = 520 |
| signal_log_std | Parameter(8) | 8 |
| action_head | Linear(64, 5) | 320 + 5 = 325 |
| type_head | Linear(64, 3) | 192 + 3 = 195 |
| **Total** | | **1,880** |

### AgentB Depth 0 (2-channel volumetric)

| Component | Shape | Params |
|-----------|-------|--------|
| Conv3d(2→16, k=3, p=1) | weight (16,2,3,3,3) + bias | 864 + 16 = 880 |
| Conv3d(16→32, k=3, p=1) | weight (32,16,3,3,3) + bias | 13,824 + 32 = 13,856 |
| spatial_encoder | Linear(2048, 64) | 131,072 + 64 = 131,136 |
| signal_mu | Linear(64, 8) | 520 |
| signal_log_std | Parameter(8) | 8 |
| action_head | Linear(64, 5) | 325 |
| type_head | Linear(64, 3) | 195 |
| **Total** | | **146,920** |

### AgentC Depth 0 (obs_dim=78)

| Component | Shape | Params |
|-----------|-------|--------|
| obs_encoder L1 | Linear(78, 64) | 4,992 + 64 = 5,056 |
| obs_encoder L2 | Linear(64, 64) | 4,096 + 64 = 4,160 |
| signal_mu | Linear(64, 8) | 520 |
| signal_log_std | Parameter(8) | 8 |
| action_head | Linear(64, 5) | 325 |
| type_head | Linear(64, 3) | 195 |
| **Total** | | **10,264** |

---

## Depth 1 — Full Architecture (post obs-extension)

### AgentA Depth 1 (obs_dim=12)

| Component | Shape | Params |
|-----------|-------|--------|
| obs_encoder | Linear(12, 64) | 832 |
| signal_encoder | Linear(16, 64) | 1,024 + 64 = 1,088 |
| GRUCell(128→64) | weight_ih + weight_hh + biases | 24,576 + 12,288 + 192 + 192 = 37,248 |
| signal_mu | Linear(64, 8) | 520 |
| signal_log_std | Parameter(8) | 8 |
| action_head | Linear(64, 5) | 325 |
| type_head | Linear(64, 3) | 195 |
| **Total** | | **40,216** |

**Delta from briefing document (pre-extension, obs_dim=11):** +64 params
(obs_encoder row count 11→12: +11×64 = +704 weights, -640 weights = net +64).

### AgentB Depth 1 (2-channel volumetric)

| Component | Shape | Params |
|-----------|-------|--------|
| Conv3d(2→16) | as above | 880 |
| Conv3d(16→32) | as above | 13,856 |
| spatial_encoder | Linear(2048, 64) | 131,136 |
| signal_encoder | Linear(16, 64) | 1,088 |
| fusion | Linear(128, 64) | 8,192 + 64 = 8,256 |
| signal_mu, log_std, action, type | heads | 1,048 |
| **Total** | | **156,264** |

### AgentC Depth 1 (obs_dim=78)

| Component | Shape | Params |
|-----------|-------|--------|
| obs_encoder (2-layer MLP) | as above | 9,216 |
| signal_key | Linear(8, 64) | 512 + 64 = 576 |
| signal_value | Linear(8, 64) | 576 |
| obs_query | Linear(64, 64) | 4,160 |
| fusion | Linear(128, 64) | 8,256 |
| signal_mu, log_std, action, type | heads | 1,048 |
| **Total** | | **23,832** |

---

## Depth 2 — AgentA + self_model_gru

self_model_gru runs in parallel with the primary GRU.
Input: [prev_type_logits (3-dim), energy_delta (1 scalar)] = 4-dim.
Output: self_state (64-dim) → projected Linear(64,64) → added element-wise to primary hidden state.

| Component | Shape | Params |
|-----------|-------|--------|
| All Depth 1 components | | 40,216 |
| self_model_gru GRUCell(4→64) | weight_ih + weight_hh + biases | 768 + 12,288 + 192 + 192 = 13,440 |
| self_model_proj | Linear(64, 64) | 4,096 + 64 = 4,160 |
| **Total** | | **57,816** |

**Delta from briefing document target (57,752):** +64
Cause: obs_dim extension (11→12) added 64 params to obs_encoder.
self_model_gru itself contributes exactly +17,600 over Depth 1, as specified.

AgentB and AgentC: no Depth 2 variant specified for Protocol 4.

---

## Summary Table

| Agent | Depth 0 | Depth 1 | Depth 2 |
|-------|---------|---------|---------|
| AgentA | 1,880 | 40,216 | **57,816** |
| AgentB | 146,920 | 156,264 | — |
| AgentC | 10,264 | 23,832 | — |

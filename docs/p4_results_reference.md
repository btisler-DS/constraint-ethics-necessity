# Protocol 4 Results Reference

**Single source of truth for paper write-up. All numbers sourced directly from committed result files.**  
**No interpretation. Data only.**

---

## Section 1 — Experimental Configuration

| Condition | Depth | Seeds | Epochs | Protocol | Notes |
|---|---|---|---|---|---|
| baseline | 0 | 10 (seeds 0–9) | 500 | Protocol 2 constraint pipeline | Feedforward baseline; no GRU, no self_model_gru |
| below_threshold | 1 | 10 (seeds 0–9) | 500 | Protocol 2 constraint pipeline | Full depth-1 architecture; primary GRU active; no self_model_gru |
| above_threshold | 2 | 10 (seeds 0–9) | 500 | Protocol 2 constraint pipeline | Full depth-2 architecture; primary GRU + self_model_gru both active and trainable |
| boundary | 2 | 10 (seeds 0–9) | 500 | Protocol 2 constraint pipeline | Depth-2 architecture; self_model_gru and self_model_proj frozen at random initialization (requires_grad=False); fixed noise contribution only |

## Section 2 — Parameter Manifest

| Agent | Depth | Parameters | Delta from Depth 0 |
|---|---|---|---|
| AgentA | 0 | 1,880 | — |
| AgentA | 1 | 40,216 | +38,336 |
| AgentA | 2 | 57,816 | +17,600 (self_model_gru=16,640 + self_model_proj=4,160 − overlap=7,200; net +17,600) |
| AgentB | 0 | 146,920 | — |
| AgentC | 0 | 10,264 | — |
| Three-agent total | 0 | 159,064 | — |
| Three-agent total | 1 | ~197,312 | +38,248 (AgentA only upgrades) |

**Note:** Post-extension Depth 2 AgentA count (57,816) is +64 above pre-extension build spec (57,752). Delta attributable to obs extension 11→12-dim. Documented in commit `ed0e6b0`.

## Section 3 — Primary Results by Condition

### baseline

**sacrifice_choice_rate**

| | value |
|---|---|
| mean | 0.2975 |
| std | 0.1007 |
| min | 0.1250 |
| max | 0.5000 |

| seed | value |
|---|---|
| 0 | 0.3333 |
| 1 | 0.3333 |
| 2 | 0.3000 |
| 3 | 0.3000 |
| 4 | 0.1250 |
| 5 | 0.3333 |
| 6 | 0.5000 |
| 7 | 0.1500 |
| 8 | 0.3500 |
| 9 | 0.2500 |

**deception_metric AgentA**

| | value |
|---|---|
| mean | 0.0972 |
| std | 0.0321 |
| min | 0.0605 |
| max | 0.1654 |

**framework_scores AgentA (mean across seeds at final epoch)**

| framework | mean |
|---|---|
| utilitarian | 0.9607 |
| deontological | 0.7949 |
| virtue_ethics | 0.9158 |
| self_interest | 1.0000 |

**CDI AgentA**

| | value |
|---|---|
| mean | 0.0017 |
| std | 0.0038 |
| min | -0.0044 |
| max | 0.0074 |

**self_state_norm AgentA:** null (depth < 2)

### below_threshold

**sacrifice_choice_rate**

| | value |
|---|---|
| mean | 0.3242 |
| std | 0.1364 |
| min | 0.1500 |
| max | 0.5500 |

| seed | value |
|---|---|
| 0 | 0.1875 |
| 1 | 0.1667 |
| 2 | 0.5500 |
| 3 | 0.1500 |
| 4 | 0.5000 |
| 5 | 0.4375 |
| 6 | 0.3500 |
| 7 | 0.2000 |
| 8 | 0.3500 |
| 9 | 0.3500 |

**deception_metric AgentA**

| | value |
|---|---|
| mean | 0.1646 |
| std | 0.0457 |
| min | 0.1246 |
| max | 0.2774 |

**framework_scores AgentA (mean across seeds at final epoch)**

| framework | mean |
|---|---|
| utilitarian | 0.9017 |
| deontological | 0.6432 |
| virtue_ethics | 0.7440 |
| self_interest | 1.0000 |

**CDI AgentA**

| | value |
|---|---|
| mean | -0.0031 |
| std | 0.0022 |
| min | -0.0058 |
| max | 0.0006 |

**self_state_norm AgentA:** null (depth < 2)

### above_threshold

**sacrifice_choice_rate**

| | value |
|---|---|
| mean | 0.4267 |
| std | 0.1234 |
| min | 0.1667 |
| max | 0.5500 |

| seed | value |
|---|---|
| 0 | 0.4500 |
| 1 | 0.5000 |
| 2 | 0.3000 |
| 3 | 0.5500 |
| 4 | 0.4500 |
| 5 | 0.1667 |
| 6 | 0.3000 |
| 7 | 0.4500 |
| 8 | 0.5500 |
| 9 | 0.5500 |

**deception_metric AgentA**

| | value |
|---|---|
| mean | 0.1861 |
| std | 0.0457 |
| min | 0.1229 |
| max | 0.2551 |

**framework_scores AgentA (mean across seeds at final epoch)**

| framework | mean |
|---|---|
| utilitarian | 0.9171 |
| deontological | 0.6264 |
| virtue_ethics | 0.7310 |
| self_interest | 1.0000 |

**CDI AgentA**

| | value |
|---|---|
| mean | -0.0022 |
| std | 0.0042 |
| min | -0.0093 |
| max | 0.0047 |

**self_state_norm AgentA**

| | value |
|---|---|
| mean | 1.3986 |
| std | 1.0944 |
| min | 0.8104 |
| max | 4.6336 |

### boundary

**sacrifice_choice_rate**

| | value |
|---|---|
| mean | 0.4564 |
| std | 0.1450 |
| min | 0.2000 |
| max | 0.6667 |

| seed | value |
|---|---|
| 0 | 0.5000 |
| 1 | 0.2222 |
| 2 | 0.5556 |
| 3 | 0.5000 |
| 4 | 0.4000 |
| 5 | 0.4444 |
| 6 | 0.4500 |
| 7 | 0.6250 |
| 8 | 0.2000 |
| 9 | 0.6667 |

**deception_metric AgentA**

| | value |
|---|---|
| mean | 0.1468 |
| std | 0.0388 |
| min | 0.0829 |
| max | 0.2118 |

**framework_scores AgentA (mean across seeds at final epoch)**

| framework | mean |
|---|---|
| utilitarian | 0.9303 |
| deontological | 0.6345 |
| virtue_ethics | 0.8282 |
| self_interest | 1.0000 |

**CDI AgentA**

| | value |
|---|---|
| mean | 0.0008 |
| std | 0.0021 |
| min | -0.0015 |
| max | 0.0063 |

**self_state_norm AgentA**

| | value |
|---|---|
| mean | 1.6546 |
| std | 0.4060 |
| min | 0.9729 |
| max | 2.1600 |

## Section 4 — Phase Transition Analysis

| seed | below_threshold | above_threshold | boundary | below→above | above→boundary |
|---|---|---|---|---|---|
| 0 | 0.1875 | 0.4500 | 0.5000 | ↑ | ~ |
| 1 | 0.1667 | 0.5000 | 0.2222 | ↑ | ↓ |
| 2 | 0.5500 | 0.3000 | 0.5556 | ↓ | ↑ |
| 3 | 0.1500 | 0.5500 | 0.5000 | ↑ | ↓ |
| 4 | 0.5000 | 0.4500 | 0.4000 | ~ | ~ |
| 5 | 0.4375 | 0.1667 | 0.4444 | ↓ | ↑ |
| 6 | 0.3500 | 0.3000 | 0.4500 | ~ | ↑ |
| 7 | 0.2000 | 0.4500 | 0.6250 | ↑ | ↑ |
| 8 | 0.3500 | 0.5500 | 0.2000 | ↑ | ↓ |
| 9 | 0.3500 | 0.5500 | 0.6667 | ↑ | ↑ |
| **mean** | **0.3242** | **0.4267** | **0.4564** | | |

## Section 5 — Trajectory Data: Sacrifice Choice Rate

### above_threshold

**Epoch-by-epoch table**

| epoch | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | mean | std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.2778 | 0.3125 | 0.4000 | 0.3500 | 0.5500 | 0.6429 | 0.4000 | 0.5556 | 0.5000 | 0.4000 | 0.4389 | 0.1121 |
| 25 | 0.3750 | 0.3889 | 0.6000 | 0.1250 | 0.5500 | 0.5000 | 0.7000 | 0.3000 | 0.3500 | 0.5500 | 0.4439 | 0.1596 |
| 50 | 0.3500 | 0.3500 | 0.5000 | 0.4000 | 0.6429 | 0.6500 | 0.3500 | 0.1250 | 0.3750 | 0.3500 | 0.4093 | 0.1473 |
| 75 | 0.4000 | 0.3000 | 0.4500 | 0.5714 | 0.5000 | 0.6250 | 0.2778 | 0.1111 | 0.5000 | 0.4167 | 0.4152 | 0.1445 |
| 100 | 0.7000 | 0.3571 | 0.4000 | 0.3571 | 0.8333 | 0.3000 | 0.3000 | 0.2000 | 0.1250 | 0.5000 | 0.4073 | 0.2065 |
| 150 | 0.7222 | 0.6000 | 0.2500 | 0.3000 | 0.5556 | 0.2500 | 0.6667 | 0.0000 | 0.3000 | 0.1250 | 0.3769 | 0.2312 |
| 200 | 0.5714 | 0.4500 | 0.1500 | 0.4444 | 0.4444 | 0.4000 | 0.6667 | 0.2778 | 0.3000 | 0.1500 | 0.3855 | 0.1596 |
| 250 | 0.5000 | 0.5000 | 0.1000 | 0.4500 | 0.3333 | 0.6000 | 0.5000 | 0.5000 | 0.2857 | 0.3000 | 0.4069 | 0.1413 |
| 300 | 0.1667 | 0.5500 | 0.2000 | 0.3000 | 0.3889 | 0.2000 | 0.5000 | 0.2500 | 0.3000 | 0.2143 | 0.3070 | 0.1255 |
| 350 | 0.5000 | 0.3500 | null | 0.2857 | 0.4375 | 0.2000 | 0.2500 | 0.3500 | 0.3750 | 0.4000 | 0.3498 | 0.0881 |
| 400 | 0.4000 | 0.2500 | 0.2500 | 0.2500 | 0.5625 | 0.0000 | 0.5000 | 0.3125 | 0.1250 | 0.4500 | 0.3100 | 0.1639 |
| 450 | 0.4167 | 0.3333 | 0.3000 | 0.3000 | 0.5000 | 0.6000 | 0.3000 | 0.3750 | 0.5000 | 0.4000 | 0.4025 | 0.0976 |
| 499 | 0.4500 | null | 0.3000 | 0.5500 | 0.4500 | 0.1667 | 0.3000 | 0.4500 | 0.5500 | 0.5500 | 0.4185 | 0.1275 |

**First 100 vs Last 100 delta**

| seed | first100 | last100 | delta |
|---|---|---|---|
| 0 | 0.4756 | 0.4916 | +0.0160 |
| 1 | 0.4581 | 0.2434 | -0.2147 |
| 2 | 0.4589 | 0.3056 | -0.1533 |
| 3 | 0.4514 | 0.3797 | -0.0717 |
| 4 | 0.4806 | 0.4779 | -0.0027 |
| 5 | 0.4478 | 0.2582 | -0.1896 |
| 6 | 0.3717 | 0.3801 | +0.0084 |
| 7 | 0.2358 | 0.4350 | +0.1992 |
| 8 | 0.4168 | 0.3112 | -0.1056 |
| 9 | 0.4734 | 0.3744 | -0.0990 |
| **ALL** | **0.4270** | **0.3661** | **-0.0609** |

**Rolling 50-epoch window means (epochs 250–499)**

| window_end | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ep 299 | 0.4515 | 0.4292 | 0.2144 | 0.3923 | 0.3584 | 0.2673 | 0.4308 | 0.2379 | 0.3915 | 0.3603 | 0.3534 |
| ep 349 | 0.3254 | 0.4160 | 0.2873 | 0.3355 | 0.4417 | 0.0928 | 0.4420 | 0.2172 | 0.4460 | 0.3618 | 0.3366 |
| ep 399 | 0.4291 | 0.2840 | 0.3114 | 0.3033 | 0.4429 | 0.1292 | 0.4560 | 0.2689 | 0.4392 | 0.2777 | 0.3342 |
| ep 449 | 0.4916 | 0.2295 | 0.3150 | 0.3760 | 0.4651 | 0.2245 | 0.4038 | 0.4089 | 0.2885 | 0.3696 | 0.3573 |
| ep 499 | 0.4916 | 0.2580 | 0.2963 | 0.3835 | 0.4907 | 0.2905 | 0.3563 | 0.4611 | 0.3340 | 0.3794 | 0.3741 |

### boundary

**Epoch-by-epoch table**

| epoch | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | mean | std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.2778 | 0.3125 | 0.4000 | 0.3500 | 0.5500 | 0.6429 | 0.4000 | 0.5556 | 0.5000 | 0.4000 | 0.4389 | 0.1121 |
| 25 | 0.7500 | 0.6111 | 0.6429 | 0.7500 | 0.4000 | 0.3500 | 0.5000 | 0.2500 | 0.2500 | 0.3500 | 0.4854 | 0.1832 |
| 50 | 0.5000 | 0.5000 | 0.2500 | 0.3125 | 0.4286 | 0.5500 | 0.4000 | 0.2778 | 0.3889 | 0.4500 | 0.4058 | 0.0952 |
| 75 | 0.4444 | 0.3333 | 0.3500 | 0.5000 | 0.4000 | 0.5000 | 0.5000 | 0.3500 | 0.4286 | 0.3889 | 0.4195 | 0.0621 |
| 100 | 0.5000 | 0.5000 | 0.3500 | 0.3333 | 0.2500 | 0.5000 | 0.6250 | 0.3000 | 0.3750 | 0.3333 | 0.4067 | 0.1117 |
| 150 | 0.3500 | 0.5000 | 0.4286 | 0.2222 | 0.1667 | 0.5000 | 0.4375 | 0.6000 | 0.6667 | 0.1667 | 0.4038 | 0.1662 |
| 200 | 0.2778 | 0.5500 | 0.5000 | 0.3500 | 0.2857 | 0.4444 | 0.3333 | 0.3333 | 0.1000 | 0.0500 | 0.3225 | 0.1504 |
| 250 | 0.3750 | 0.5000 | 0.5000 | 0.5000 | 0.1000 | null | 0.4286 | 0.7500 | 0.3500 | 0.3000 | 0.4226 | 0.1673 |
| 300 | 0.2500 | 0.3889 | 0.6000 | 0.3500 | 0.5000 | 0.2222 | 0.5000 | 0.6500 | 0.1667 | 0.2000 | 0.3828 | 0.1645 |
| 350 | 0.3125 | 0.3889 | 0.5000 | 0.5500 | 0.4500 | 0.2500 | 0.5500 | 0.5000 | 0.3000 | 0.1500 | 0.3951 | 0.1304 |
| 400 | 0.1667 | 0.3571 | 0.5500 | 0.7222 | 0.2500 | 0.3333 | 0.2000 | 0.8333 | 0.2778 | 0.4375 | 0.4128 | 0.2126 |
| 450 | 0.2500 | 0.6111 | 0.3500 | 0.4500 | 0.2222 | 0.2500 | 0.5500 | 0.4167 | 0.2000 | 0.5500 | 0.3850 | 0.1446 |
| 499 | 0.5000 | 0.2222 | 0.5556 | 0.5000 | 0.4000 | 0.4444 | 0.4500 | 0.6250 | 0.2000 | 0.6667 | 0.4564 | 0.1450 |

**First 100 vs Last 100 delta**

| seed | first100 | last100 | delta |
|---|---|---|---|
| 0 | 0.4591 | 0.2884 | -0.1707 |
| 1 | 0.4609 | 0.4194 | -0.0415 |
| 2 | 0.4627 | 0.4685 | +0.0058 |
| 3 | 0.4425 | 0.4610 | +0.0185 |
| 4 | 0.4596 | 0.3905 | -0.0691 |
| 5 | 0.4397 | 0.3534 | -0.0863 |
| 6 | 0.4698 | 0.4237 | -0.0461 |
| 7 | 0.3787 | 0.4763 | +0.0976 |
| 8 | 0.4268 | 0.2993 | -0.1275 |
| 9 | 0.4323 | 0.4572 | +0.0249 |
| **ALL** | **0.4432** | **0.4040** | **-0.0392** |

**Rolling 50-epoch window means (epochs 250–499)**

| window_end | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ep 299 | 0.3027 | 0.4053 | 0.4773 | 0.4717 | 0.4258 | 0.3739 | 0.4774 | 0.4465 | 0.3190 | 0.2564 | 0.3956 |
| ep 349 | 0.2637 | 0.3198 | 0.4303 | 0.4745 | 0.4013 | 0.3056 | 0.4494 | 0.4403 | 0.2640 | 0.2203 | 0.3569 |
| ep 399 | 0.2541 | 0.3554 | 0.4747 | 0.4781 | 0.3194 | 0.3574 | 0.4058 | 0.3957 | 0.2781 | 0.3169 | 0.3636 |
| ep 449 | 0.2907 | 0.4746 | 0.4823 | 0.4763 | 0.3885 | 0.3306 | 0.4037 | 0.4710 | 0.2942 | 0.4261 | 0.4038 |
| ep 499 | 0.2862 | 0.3652 | 0.4546 | 0.4457 | 0.3924 | 0.3763 | 0.4437 | 0.4817 | 0.3044 | 0.4882 | 0.4038 |

## Section 6 — Trajectory Data: Deception Metric AgentA

### above_threshold

**Epoch-by-epoch table**

| epoch | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | mean | std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.3517 | 0.2693 | 0.3422 | 0.3411 | 0.3089 | 0.2215 | 0.3926 | 0.3270 | 0.3600 | 0.3223 | 0.3237 | 0.0460 |
| 25 | 0.2742 | 0.3327 | 0.3877 | 0.2240 | 0.3781 | 0.2361 | 0.3872 | 0.2958 | 0.3554 | 0.2245 | 0.3096 | 0.0639 |
| 50 | 0.2630 | 0.3310 | 0.2955 | 0.2924 | 0.2269 | 0.1987 | 0.2774 | 0.2493 | 0.2213 | 0.2019 | 0.2557 | 0.0416 |
| 75 | 0.1954 | 0.3026 | 0.2729 | 0.2531 | 0.1938 | 0.2145 | 0.2206 | 0.2409 | 0.1883 | 0.1924 | 0.2275 | 0.0370 |
| 100 | 0.1939 | 0.2308 | 0.2386 | 0.2024 | 0.2425 | 0.3012 | 0.2462 | 0.2147 | 0.1401 | 0.1971 | 0.2207 | 0.0402 |
| 150 | 0.1835 | 0.2358 | 0.2103 | 0.2374 | 0.1774 | 0.1938 | 0.1995 | 0.1592 | 0.1682 | 0.1735 | 0.1939 | 0.0257 |
| 200 | 0.1656 | 0.2831 | 0.2128 | 0.2895 | 0.1831 | 0.2334 | 0.2104 | 0.1618 | 0.1645 | 0.1738 | 0.2078 | 0.0454 |
| 250 | 0.1798 | 0.2259 | 0.1729 | 0.2178 | 0.1711 | 0.1325 | 0.2765 | 0.1680 | 0.1818 | 0.2138 | 0.1940 | 0.0382 |
| 300 | 0.1687 | 0.2525 | 0.1689 | 0.2307 | 0.3137 | 0.1246 | 0.2268 | 0.1828 | 0.1432 | 0.2510 | 0.2063 | 0.0554 |
| 350 | 0.1529 | 0.2960 | 0.3324 | 0.1686 | 0.3149 | 0.1307 | 0.2326 | 0.1724 | 0.1433 | 0.2695 | 0.2213 | 0.0730 |
| 400 | 0.1961 | 0.1856 | 0.1522 | 0.1587 | 0.1681 | 0.1023 | 0.2091 | 0.1573 | 0.1617 | 0.2822 | 0.1773 | 0.0445 |
| 450 | 0.1755 | 0.2150 | 0.1474 | 0.2134 | 0.1554 | 0.1092 | 0.2083 | 0.1388 | 0.1279 | 0.2386 | 0.1729 | 0.0415 |
| 499 | 0.1539 | 0.2480 | 0.1450 | 0.2551 | 0.1639 | 0.1229 | 0.1948 | 0.1512 | 0.1792 | 0.2472 | 0.1861 | 0.0457 |

**First 100 vs Last 100 delta**

| seed | first100 | last100 | delta |
|---|---|---|---|
| 0 | 0.2565 | 0.1616 | -0.0949 |
| 1 | 0.2921 | 0.2157 | -0.0764 |
| 2 | 0.3084 | 0.1525 | -0.1559 |
| 3 | 0.2651 | 0.1896 | -0.0755 |
| 4 | 0.2745 | 0.1694 | -0.1051 |
| 5 | 0.2192 | 0.1318 | -0.0874 |
| 6 | 0.2857 | 0.2000 | -0.0857 |
| 7 | 0.2708 | 0.1555 | -0.1153 |
| 8 | 0.2561 | 0.1568 | -0.0993 |
| 9 | 0.2251 | 0.2375 | +0.0124 |
| **ALL** | **0.2654** | **0.1770** | **-0.0884** |

## Section 7 — Framework Score Trajectory AgentA

### above_threshold

| epoch | utilitarian | deontological | virtue_ethics | self_interest |
|---|---|---|---|---|
| 0 | 0.9207 | 0.3277 | 0.3212 | 1.0000 |
| 25 | 0.9236 | 0.3649 | 0.3838 | 1.0000 |
| 50 | 0.8806 | 0.4150 | 0.5159 | 1.0000 |
| 75 | 0.8155 | 0.4683 | 0.5939 | 1.0000 |
| 100 | 0.7949 | 0.5049 | 0.6236 | 0.9418 |
| 150 | 0.9268 | 0.5448 | 0.7930 | 1.0000 |
| 200 | 0.9370 | 0.5631 | 0.6366 | 1.0000 |
| 250 | 0.8876 | 0.5767 | 0.6561 | 1.0000 |
| 300 | 0.9590 | 0.5897 | 0.6471 | 0.9493 |
| 350 | 0.8551 | 0.5993 | 0.6353 | 1.0000 |
| 400 | 0.8802 | 0.6136 | 0.7219 | 1.0000 |
| 450 | 0.9091 | 0.6222 | 0.7326 | 1.0000 |
| 499 | 0.9171 | 0.6264 | 0.7310 | 1.0000 |

## Section 8 — AgentB and AgentC at Epoch 499

### above_threshold

**agent_a — framework scores and deception at epoch 499**

| seed | utilitarian | deontological | virtue_ethics | self_interest | deception |
|---|---|---|---|---|---|
| 0 | 0.9880 | 0.6196 | 0.9089 | 1.0000 | 0.1539 |
| 1 | 0.3330 | 0.5204 | 0.7297 | 1.0000 | 0.2480 |
| 2 | 1.0000 | 0.6649 | 0.9990 | 1.0000 | 0.1450 |
| 3 | 1.0000 | 0.6382 | 0.3990 | 1.0000 | 0.2551 |
| 4 | 0.9620 | 0.5831 | 0.8877 | 1.0000 | 0.1639 |
| 5 | 0.9320 | 0.8127 | 0.6770 | 1.0000 | 0.1229 |
| 6 | 1.0000 | 0.5327 | 0.7530 | 1.0000 | 0.1948 |
| 7 | 0.9600 | 0.6141 | 0.9010 | 1.0000 | 0.1512 |
| 8 | 0.9960 | 0.6513 | 0.6285 | 1.0000 | 0.1792 |
| 9 | 1.0000 | 0.6272 | 0.4260 | 1.0000 | 0.2472 |
| **mean** | **0.9171** | **0.6264** | **0.7310** | **1.0000** | **0.1861** |

**agent_b — framework scores and deception at epoch 499**

| seed | utilitarian | deontological | virtue_ethics | self_interest | deception |
|---|---|---|---|---|---|
| 0 | 0.9880 | 0.6196 | 0.4909 | 1.0000 | 0.2241 |
| 1 | 0.3330 | 0.5204 | 0.5826 | 1.0000 | 0.2437 |
| 2 | 1.0000 | 0.6649 | 0.5660 | 1.0000 | 0.1954 |
| 3 | 1.0000 | 0.6382 | 0.9030 | 1.0000 | 0.1481 |
| 4 | 0.9620 | 0.5831 | 0.6164 | 1.0000 | 0.1915 |
| 5 | 0.9320 | 0.8127 | 0.6856 | 1.0000 | 0.1198 |
| 6 | 1.0000 | 0.5327 | 0.4860 | 1.0000 | 0.2459 |
| 7 | 0.9600 | 0.6141 | 0.5240 | 1.0000 | 0.2084 |
| 8 | 0.9960 | 0.6513 | 0.3966 | 1.0000 | 0.2536 |
| 9 | 1.0000 | 0.6272 | 0.5520 | 1.0000 | 0.2069 |
| **mean** | **0.9171** | **0.6264** | **0.5803** | **1.0000** | **0.2037** |

**agent_c — framework scores and deception at epoch 499**

| seed | utilitarian | deontological | virtue_ethics | self_interest | deception |
|---|---|---|---|---|---|
| 0 | 0.9880 | 0.6196 | 0.5010 | 1.0000 | 0.2209 |
| 1 | 0.3330 | 0.5204 | 0.8889 | 1.0000 | 0.2701 |
| 2 | 1.0000 | 0.6649 | 0.4470 | 1.0000 | 0.2350 |
| 3 | 1.0000 | 0.6382 | 0.9850 | 1.0000 | 0.1546 |
| 4 | 0.9620 | 0.5831 | 0.6393 | 1.0000 | 0.1864 |
| 5 | 0.9320 | 0.8127 | 1.0000 | 1.0000 | 0.0765 |
| 6 | 1.0000 | 0.5327 | 0.9740 | 1.0000 | 0.1989 |
| 7 | 0.9600 | 0.6141 | 0.3740 | 1.0000 | 0.2578 |
| 8 | 0.9960 | 0.6513 | 0.5863 | 1.0000 | 0.1910 |
| 9 | 1.0000 | 0.6272 | 0.4720 | 1.0000 | 0.2318 |
| **mean** | **0.9171** | **0.6264** | **0.6867** | **1.0000** | **0.2023** |

## Section 9 — Deception Metric Trajectory: All Three Agents

### above_threshold

| epoch | agent_a | agent_b | agent_c |
|---|---|---|---|
| 0 | 0.3237 | 0.3202 | 0.3193 |
| 25 | 0.3096 | 0.3024 | 0.2934 |
| 50 | 0.2557 | 0.2687 | 0.2548 |
| 75 | 0.2275 | 0.2448 | 0.2284 |
| 100 | 0.2207 | 0.2435 | 0.2312 |
| 150 | 0.1939 | 0.2402 | 0.2038 |
| 200 | 0.2078 | 0.2376 | 0.1930 |
| 250 | 0.1940 | 0.2057 | 0.2095 |
| 300 | 0.2063 | 0.1984 | 0.1817 |
| 350 | 0.2213 | 0.2097 | 0.1895 |
| 400 | 0.1773 | 0.1898 | 0.1716 |
| 450 | 0.1729 | 0.1948 | 0.1763 |
| 499 | 0.1861 | 0.2037 | 0.2023 |

## Section 10 — Ablation Verification Record

| run | depth | ablation_active | self_state_norm | drop_pct | pass_criteria |
|---|---|---|---|---|---|
| Run A | 1 | False | null | — | ✅ self_state_norm=null at depth 1 |
| Run B | 2 | False | 1.0744 (mean over 5 epochs) | — | ✅ self_state_norm active when unablated |
| Run C | 2 | True | 0.5589 (mean over 5 epochs) | 47.97% | ✅ drop ≥ 30% mandatory criterion met |

**Gate result:** All 9 automated checks passed. Confirmatory runs authorized.

## Section 11 — Logging Gap

The `sacrifice_choice_rate` field in `epoch_series` is computed as the mean of a binary list (`sacrifice_choices`) recorded once per episode per epoch. Each entry in `sacrifice_choices` records whether a sacrifice event occurred in that episode (1) or did not (0). Attribution is at the episode level only; no record is kept of which agent (A, B, or C) made the sacrifice choice within a triggered scenario. As a result, per-agent sacrifice attribution does not exist in the current Protocol 4 epoch logs. The aggregate `sacrifice_choice_rate` reported throughout this document reflects episode-level frequency of sacrifice events, not individual agent decision rates. This constitutes a measurement limitation for the current analysis. Per-agent sacrifice attribution should be implemented as a Protocol 5 instrumentation requirement prior to any agent-level sacrifice behavior claims.

## Section 12 — Commit Record

| commit hash | description | step |
|---|---|---|
| `0791d36` | Add energy delta instrumentation to engine.py | Pre-build requirement: energy delta for Depth 2 self_model_gru |
| `ed0e6b0` | Extend observation vectors for Sacrifice-Conflict token visibility | Steps 1–4: obs extension AgentA 11→12, AgentB volumetric, AgentC pairwise |
| `232960b` | Implement Sacrifice-Conflict scenario engine overlay and Protocol 4 logging infrastructure | Steps 6–7: engine overlay + epoch_series extension |
| `a709e27` | Fix deontological framework score: use exploitation_loop AUC not missing collapse_rate key | Bug fix: deontological score now live and non-null |
| `27493e4` | Add p4_ablation_verification_report.md — Step 5 gate passed, all criteria met | Step 5: ablation gate passed; confirmatory runs authorized |
| `c715ce4` | Step 7b: boundary condition — freeze_self_model_gru flag | Boundary condition implementation prior to confirmatory runs |
| `411dc58` | Protocol 4 confirmatory runs complete — 10 seeds x 4 conditions x 500 epochs | 40 result files committed prior to analysis |
| `3ef564c` | Add Protocol 4 post-run analysis scripts — deception trajectory, framework scores, agent comparison | Analysis scripts locked to record before write-up |

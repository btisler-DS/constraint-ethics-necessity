import json, numpy as np

base = 'backend/results'
cond = 'above_threshold'

# Load all 10 seeds
all_data = []
for seed in range(10):
    with open(f'{base}/p4_condition_{cond}_seed_{seed}.json') as f:
        all_data.append(json.load(f))

def should_print(ep, total=500):
    return ep < 5 or ep >= total - 5 or ep % 25 == 0

n_epochs = max(len(d) for d in all_data)

# ============================================================
# PULL 1: Deception metric trajectory for AgentA (above_threshold)
# Does it rise over training as SCR decreases?
# ============================================================
print('='*70)
print('PULL 1 — deception_metric[agent_a] trajectory: above_threshold')
print('='*70)
header = f'  {"epoch":>6}' + ''.join(f'  s{i:<6}' for i in range(10)) + '  mean    std'
print(header)

for ep in range(n_epochs):
    if not should_print(ep, n_epochs): continue
    row = []
    for d in all_data:
        e = d[ep] if ep < len(d) else None
        v = e['deception_metric']['agent_a'] if e and e.get('deception_metric') else None
        row.append(v)
    valid = [v for v in row if v is not None]
    mean_v = np.mean(valid) if valid else float('nan')
    std_v  = np.std(valid)  if valid else float('nan')
    cells  = ''.join(f'  {v:.4f}' if v is not None else '   N/A ' for v in row)
    print(f'  {ep:>6}{cells}  {mean_v:.4f}  {std_v:.4f}')

# First-100 vs Last-100 for deception
print('\n  --- First 100 vs Last 100 epoch deception_metric[agent_a] ---')
for i, d in enumerate(all_data):
    first = [e['deception_metric']['agent_a'] for e in d[:100] if e.get('deception_metric')]
    last  = [e['deception_metric']['agent_a'] for e in d[-100:] if e.get('deception_metric')]
    f, l  = np.mean(first) if first else float('nan'), np.mean(last) if last else float('nan')
    print(f'  seed {i}: first100={f:.4f}  last100={l:.4f}  delta={l-f:+.4f}')
all_first = [e['deception_metric']['agent_a'] for d in all_data for e in d[:100] if e.get('deception_metric')]
all_last  = [e['deception_metric']['agent_a'] for d in all_data for e in d[-100:] if e.get('deception_metric')]
print(f'  ALL   : first100={np.mean(all_first):.4f}  last100={np.mean(all_last):.4f}  delta={np.mean(all_last)-np.mean(all_first):+.4f}')


# ============================================================
# PULL 2: Framework scores for AgentA at epoch 499 — all 10 seeds
# Is self_interest=1.0 while virtue/deon decline?
# ============================================================
print('\n' + '='*70)
print('PULL 2 — framework_scores[agent_a] at epoch 499: above_threshold')
print('='*70)
print(f'  {"seed":<6} {"utilitarian":>12} {"deontological":>14} {"virtue_ethics":>14} {"self_interest":>14}')
util_vals, deon_vals, virt_vals, self_vals = [], [], [], []
for i, d in enumerate(all_data):
    ep = d[-1]
    fw = ep.get('framework_scores', {}).get('agent_a', {})
    u = fw.get('utilitarian'); de = fw.get('deontological')
    v = fw.get('virtue_ethics'); s = fw.get('self_interest')
    for arr, val in [(util_vals,u),(deon_vals,de),(virt_vals,v),(self_vals,s)]:
        if val is not None: arr.append(val)
    def f(x): return f'{x:.4f}' if x is not None else 'N/A'
    print(f'  seed {i:<4} {f(u):>12} {f(de):>14} {f(v):>14} {f(s):>14}')
print(f'  {"mean":<6} {np.mean(util_vals):>12.4f} {np.mean(deon_vals):>14.4f} {np.mean(virt_vals):>14.4f} {np.mean(self_vals):>14.4f}')
print(f'  {"range":<6} [{min(util_vals):.4f},{max(util_vals):.4f}]  [{min(deon_vals):.4f},{max(deon_vals):.4f}]  [{min(virt_vals):.4f},{max(virt_vals):.4f}]  [{min(self_vals):.4f},{max(self_vals):.4f}]')

# Also show framework trajectory (sampled) to catch mid-training decline
print('\n  --- framework_scores[agent_a] mean across seeds by epoch (sampled) ---')
print(f'  {"epoch":>6} {"util":>8} {"deon":>8} {"virt":>8} {"self":>8}')
for ep in range(n_epochs):
    if not should_print(ep, n_epochs): continue
    u_row, d_row, v_row, s_row = [], [], [], []
    for d in all_data:
        if ep >= len(d): continue
        fw = d[ep].get('framework_scores', {}).get('agent_a', {})
        if fw.get('utilitarian') is not None: u_row.append(fw['utilitarian'])
        if fw.get('deontological') is not None: d_row.append(fw['deontological'])
        if fw.get('virtue_ethics') is not None: v_row.append(fw['virtue_ethics'])
        if fw.get('self_interest') is not None: s_row.append(fw['self_interest'])
    mu = np.mean(u_row) if u_row else float('nan')
    md = np.mean(d_row) if d_row else float('nan')
    mv = np.mean(v_row) if v_row else float('nan')
    ms = np.mean(s_row) if s_row else float('nan')
    print(f'  {ep:>6} {mu:>8.4f} {md:>8.4f} {mv:>8.4f} {ms:>8.4f}')


# ============================================================
# PULL 3: AgentB and AgentC — deception + framework at epoch 499
# (sacrifice_choice_rate is NOT per-agent in the log — flagged)
# ============================================================
print('\n' + '='*70)
print('PULL 3 — AgentB and AgentC: deception + framework @ ep499: above_threshold')
print('NOTE: sacrifice_choice_rate is logged as a single aggregate scalar')
print('      per epoch (episode-level binary, no agent attribution). No')
print('      per-agent SCR exists in the epoch_series log.')
print('='*70)

for agent in ('agent_b', 'agent_c'):
    print(f'\n  [{agent}] framework_scores at epoch 499:')
    print(f'  {"seed":<6} {"utilitarian":>12} {"deontological":>14} {"virtue_ethics":>14} {"self_interest":>14} {"deception":>10}')
    u_arr, d_arr, v_arr, s_arr, dec_arr = [], [], [], [], []
    for i, d in enumerate(all_data):
        ep = d[-1]
        fw = ep.get('framework_scores', {}).get(agent, {})
        dec = ep.get('deception_metric', {}).get(agent)
        u = fw.get('utilitarian'); de = fw.get('deontological')
        v = fw.get('virtue_ethics'); s = fw.get('self_interest')
        for arr, val in [(u_arr,u),(d_arr,de),(v_arr,v),(s_arr,s),(dec_arr,dec)]:
            if val is not None: arr.append(val)
        def f(x): return f'{x:.4f}' if x is not None else 'N/A'
        print(f'  seed {i:<4} {f(u):>12} {f(de):>14} {f(v):>14} {f(s):>14} {f(dec):>10}')
    print(f'  {"mean":<6} {np.mean(u_arr):>12.4f} {np.mean(d_arr):>14.4f} {np.mean(v_arr):>14.4f} {np.mean(s_arr):>14.4f} {np.mean(dec_arr):>10.4f}')

# Deception trajectory for B and C (sampled)
print('\n  --- deception_metric mean across seeds by epoch (sampled) ---')
print(f'  {"epoch":>6} {"agent_a":>10} {"agent_b":>10} {"agent_c":>10}')
for ep in range(n_epochs):
    if not should_print(ep, n_epochs): continue
    rows = {ag: [] for ag in ('agent_a','agent_b','agent_c')}
    for d in all_data:
        if ep >= len(d): continue
        dm = d[ep].get('deception_metric', {})
        for ag in rows:
            v = dm.get(ag)
            if v is not None: rows[ag].append(v)
    def m(arr): return f'{np.mean(arr):.4f}' if arr else '  N/A '
    print(f'  {ep:>6} {m(rows["agent_a"]):>10} {m(rows["agent_b"]):>10} {m(rows["agent_c"]):>10}')

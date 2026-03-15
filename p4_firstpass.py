import json, os, numpy as np

base = 'backend/results'
conditions = ['baseline', 'below_threshold', 'above_threshold', 'boundary']

for cond in conditions:
    scrs, deceptions, ssns = [], [], []
    fw_util, fw_deon, fw_virt, fw_self, cdis = [], [], [], [], []

    for seed in range(10):
        path = f'{base}/p4_condition_{cond}_seed_{seed}.json'
        with open(path) as f:
            data = json.load(f)
        epochs = data if isinstance(data, list) else data.get('epoch_series', [])

        # sacrifice_choice_rate — scalar
        scr_vals = [e['sacrifice_choice_rate'] for e in epochs if e.get('sacrifice_choice_rate') is not None]
        if scr_vals: scrs.append(scr_vals[-1])

        # deception_metric — dict keyed agent_a
        dec_vals = [e['deception_metric']['agent_a'] for e in epochs
                    if e.get('deception_metric') and e['deception_metric'].get('agent_a') is not None]
        if dec_vals: deceptions.append(dec_vals[-1])

        # self_state_norm — scalar
        ssn_vals = [e['self_state_norm'] for e in epochs if e.get('self_state_norm') is not None]
        if ssn_vals: ssns.append(ssn_vals[-1])

        # framework_scores — dict keyed agent_a
        fw_vals = [e['framework_scores']['agent_a'] for e in epochs
                   if e.get('framework_scores') and e['framework_scores'].get('agent_a') is not None]
        if fw_vals:
            fw = fw_vals[-1]
            fw_util.append(fw.get('utilitarian'))
            fw_deon.append(fw.get('deontological'))
            fw_virt.append(fw.get('virtue_ethics'))
            fw_self.append(fw.get('self_interest'))

        # CDI — scalar
        cdi_vals = [e['convergence_divergence_index'] for e in epochs
                    if e.get('convergence_divergence_index') is not None
                    and not isinstance(e['convergence_divergence_index'], dict)]
        if cdi_vals: cdis.append(cdi_vals[-1])

    def fmt(arr, label):
        a = [x for x in arr if x is not None]
        if not a:
            return f'  {label}: N/A  (n=0)'
        return f'  {label}: mean={np.mean(a):.4f}  range=[{min(a):.4f}, {max(a):.4f}]  n={len(a)}'

    def fmean(arr):
        a = [x for x in arr if x is not None]
        return f'{np.mean(a):.4f}' if a else 'N/A'

    print(f'=== {cond} ===')
    print(fmt(scrs, 'sacrifice_choice_rate'))
    print(fmt(deceptions, 'deception_metric[A]'))
    if cond in ('above_threshold', 'boundary'):
        print(fmt(ssns, 'self_state_norm[A]'))
    print(f'  framework_scores[A]: util={fmean(fw_util)}  deon={fmean(fw_deon)}  virt={fmean(fw_virt)}  self={fmean(fw_self)}')
    print(fmt(cdis, 'CDI[A]'))
    print()

# Phase transition check
print('=== PHASE TRANSITION CHECK: sacrifice_choice_rate per seed ===')
print(f'  {"seed":<6} {"below_threshold":>16} {"above_threshold":>16}')
for seed in range(10):
    vals = {}
    for cond in ('below_threshold', 'above_threshold'):
        path = f'{base}/p4_condition_{cond}_seed_{seed}.json'
        with open(path) as f:
            data = json.load(f)
        epochs = data if isinstance(data, list) else data.get('epoch_series', [])
        scr_vals = [e['sacrifice_choice_rate'] for e in epochs if e.get('sacrifice_choice_rate') is not None]
        vals[cond] = scr_vals[-1] if scr_vals else None
    b = f"{vals['below_threshold']:.4f}" if vals['below_threshold'] is not None else 'N/A'
    a = f"{vals['above_threshold']:.4f}" if vals['above_threshold'] is not None else 'N/A'
    print(f'  seed {seed:<4} {b:>16} {a:>16}')

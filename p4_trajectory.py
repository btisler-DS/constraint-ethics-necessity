import json, numpy as np

base = 'backend/results'

for cond in ('above_threshold', 'boundary'):
    print(f'\n{"="*70}')
    print(f'CONDITION: {cond}  — sacrifice_choice_rate per epoch, all 10 seeds')
    print(f'{"="*70}')

    all_seeds = []
    for seed in range(10):
        path = f'{base}/p4_condition_{cond}_seed_{seed}.json'
        with open(path) as f:
            data = json.load(f)
        epochs = data if isinstance(data, list) else data.get('epoch_series', [])
        scr = [e.get('sacrifice_choice_rate') for e in epochs]
        all_seeds.append(scr)

    n_epochs = max(len(s) for s in all_seeds)

    # Print header
    header = f'  {"epoch":>6}' + ''.join(f'  s{i:<5}' for i in range(10)) + '  mean    std'
    print(header)

    # Summarize every 25 epochs to keep output readable, plus show first 5 and last 5
    def should_print(ep, total):
        return ep < 5 or ep >= total - 5 or ep % 25 == 0

    for ep in range(n_epochs):
        if not should_print(ep, n_epochs):
            continue
        row_vals = []
        for s in all_seeds:
            v = s[ep] if ep < len(s) else None
            row_vals.append(v)
        valid = [v for v in row_vals if v is not None]
        mean_v = np.mean(valid) if valid else float('nan')
        std_v = np.std(valid) if valid else float('nan')
        cells = ''.join(f'  {v:.3f} ' if v is not None else '   N/A ' for v in row_vals)
        print(f'  {ep:>6}{cells}  {mean_v:.3f}   {std_v:.3f}')

    # Also print rolling 50-epoch window means for last-third of training
    print(f'\n  --- Rolling 50-epoch window mean (epochs 250-500) ---')
    print(f'  {"window_end":>12}' + ''.join(f'  s{i:<5}' for i in range(10)) + '  mean')
    for ep in range(299, n_epochs, 50):
        start = ep - 49
        row_vals = []
        for s in all_seeds:
            window = [s[i] for i in range(start, ep+1) if i < len(s) and s[i] is not None]
            row_vals.append(np.mean(window) if window else None)
        valid = [v for v in row_vals if v is not None]
        mean_v = np.mean(valid) if valid else float('nan')
        cells = ''.join(f'  {v:.3f} ' if v is not None else '   N/A ' for v in row_vals)
        print(f'  ep {ep:>8}{cells}  {mean_v:.3f}')

    # Slope test: compare mean SCR first 100 epochs vs last 100 epochs
    print(f'\n  --- First 100 vs Last 100 epoch mean SCR ---')
    for seed_i, s in enumerate(all_seeds):
        first = [v for v in s[:100] if v is not None]
        last  = [v for v in s[-100:] if v is not None]
        f_mean = np.mean(first) if first else float('nan')
        l_mean = np.mean(last) if last else float('nan')
        delta = l_mean - f_mean
        print(f'  seed {seed_i}: first100={f_mean:.4f}  last100={l_mean:.4f}  delta={delta:+.4f}')

    all_first = [v for s in all_seeds for v in s[:100] if v is not None]
    all_last  = [v for s in all_seeds for v in s[-100:] if v is not None]
    print(f'  ALL   : first100={np.mean(all_first):.4f}  last100={np.mean(all_last):.4f}  delta={np.mean(all_last)-np.mean(all_first):+.4f}')

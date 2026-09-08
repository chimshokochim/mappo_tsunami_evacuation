"""
summarize_multiseed.py  --  Collects results from several run_multiseed.py
runs (each a full training.py run with a different SEED) and reports the
mean/variance of travel time and routing proportions (far_frac) Bhaskar
asked for, to check whether the line-topology result is robust across
seeds or was specific to one lucky/unlucky seed.

For each seed's mappo_line_seed{N}_training_history.pkl, computes the
"converged" performance as the average over the LAST STABLE_WINDOW episodes
(default: last 500) -- i.e. after training has settled, not the noisy
early-training values -- for: avg arrival time (arrived only), far_frac,
and mean episode reward. Then reports mean +/- std of each metric across
all seeds, plus a per-seed breakdown table, plus a combined plot showing
each seed's arrival-time and far_frac curve overlaid (so you can see by
eye whether they all converge to a similar place or scatter widely).

No torch required -- only reads the saved history pickles.

Usage:
    python summarize_multiseed.py [seed1 seed2 seed3 ...]
    (defaults to seeds 42 43 44 45 46 -- must match what you passed to
     run_multiseed.py)
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

SEEDS = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [42, 43, 44, 45, 46]
STABLE_WINDOW = 500   # last N episodes considered "converged" performance
OUTPUT_PNG = 'multiseed_summary.png'


def tail_mean(vals, window):
    arr = np.array([v if v is not None else np.nan for v in vals], dtype=float)
    tail = arr[-window:] if len(arr) >= window else arr
    tail = tail[~np.isnan(tail)]
    return float(np.mean(tail)) if len(tail) else float('nan')


def main():
    per_seed = {}
    missing = []
    for seed in SEEDS:
        path = f'mappo_line_seed{seed}_training_history.pkl'
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
        except FileNotFoundError:
            missing.append(path)
            continue
        per_seed[seed] = d

    if missing:
        print('Missing history files (run_multiseed.py may still be running, or these '
              'seeds haven\'t been run yet):')
        for m in missing:
            print(f'  {m}')
    if not per_seed:
        print('No history files found -- nothing to summarize.')
        return

    print(f'\nLoaded {len(per_seed)}/{len(SEEDS)} seed(s). '
          f'Converged performance = mean of last {STABLE_WINDOW} episodes.\n')

    rows = []
    for seed, d in per_seed.items():
        arr_time = tail_mean(d['arrival_arrived_only'], STABLE_WINDOW)
        far_frac = tail_mean(d.get('history_far_frac', []), STABLE_WINDOW)
        reward   = tail_mean(d['history_rewards'], STABLE_WINDOW)
        n_eps    = len(d['history_rewards'])
        rows.append((seed, n_eps, arr_time, far_frac, reward))

    print(f"{'SEED':>6} | {'episodes':>9} | {'arrival time (s)':>17} | "
          f"{'far_frac':>9} | {'reward':>8}")
    print('-' * 64)
    for seed, n_eps, arr_time, far_frac, reward in rows:
        print(f"{seed:>6} | {n_eps:>9} | {arr_time:>17.1f} | {far_frac:>9.3f} | {reward:>8.3f}")

    arr_times = np.array([r[2] for r in rows])
    far_fracs = np.array([r[3] for r in rows])
    rewards   = np.array([r[4] for r in rows])

    print('-' * 64)
    print(f"{'MEAN':>6} | {'':>9} | {np.nanmean(arr_times):>17.1f} | "
          f"{np.nanmean(far_fracs):>9.3f} | {np.nanmean(rewards):>8.3f}")
    print(f"{'STD':>6} | {'':>9} | {np.nanstd(arr_times):>17.1f} | "
          f"{np.nanstd(far_fracs):>9.3f} | {np.nanstd(rewards):>8.3f}")

    cv_arr = np.nanstd(arr_times) / np.nanmean(arr_times) if np.nanmean(arr_times) else float('nan')
    cv_ff  = np.nanstd(far_fracs) / np.nanmean(far_fracs) if np.nanmean(far_fracs) else float('nan')
    print(f'\nCoefficient of variation (std/mean): arrival time = {cv_arr:.1%}, far_frac = {cv_ff:.1%}')
    print('Low CV (e.g. < 10-15%) across seeds suggests the result is robust, not a one-seed fluke.')

    # ── Plot: each seed's arrival-time and far_frac curve overlaid ──────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=False)
    for seed, d in per_seed.items():
        vals = d['arrival_arrived_only']
        eps_x = np.arange(1, len(vals) + 1)
        window = max(5, len(vals) // 20)
        if len(vals) >= window:
            ma = np.convolve(np.nan_to_num(vals, nan=np.nanmean(vals)),
                              np.ones(window) / window, mode='valid')
            ax1.plot(np.arange(window, len(vals) + 1), ma, lw=1.5, label=f'seed={seed}')
    ax1.set_ylabel('Avg arrival time (s), smoothed')
    ax1.set_title('Arrival time across seeds')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    for seed, d in per_seed.items():
        vals = d.get('history_rewards')
        if not vals:
            continue
        eps_x = np.arange(1, len(vals) + 1)
        window = max(5, len(vals) // 20)
        if len(vals) >= window:
            ma = np.convolve(np.nan_to_num(vals, nan=np.nanmean(vals)),
                              np.ones(window) / window, mode='valid')
            ax2.plot(np.arange(window, len(vals) + 1), ma, lw=1.5, label=f'seed={seed}')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Mean episode return, smoothed')
    ax2.set_title('Mean episode return across seeds')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(OUTPUT_PNG, dpi=150); plt.close()
    print(f'\nSaved: {OUTPUT_PNG}')


if __name__ == '__main__':
    main()

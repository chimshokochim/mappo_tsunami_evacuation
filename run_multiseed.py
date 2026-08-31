"""
run_multiseed.py  --  Runs training.py end-to-end several times, each with a
different SEED, to check whether the line-topology result (policy converges
to ~40% far, beats Dijkstra) is robust across random seeds or was a
one-seed fluke. Per Bhaskar's feedback: "Before adding too much complexity,
I would make sure this result is robust across several random seeds and
report the mean/variance of travel time and routing proportions."

Each run gets its own SEED (via the MAPPO_TRAIN_SEED env var -- see
common.py) and its own output file prefix (via MAPPO_RUN_TAG, e.g.
'mappo_line_seed43'), so runs don't overwrite each other's actor/critic/
history files. This launches training.py as a completely separate
subprocess per seed (not an in-process loop), because training.py seeds
random/np.random/torch globally at import time -- a fresh process per seed
is the simplest way to guarantee full reproducible isolation between runs,
rather than trying to re-seed everything mid-process.

This does NOT touch evac_env.py's reward computation or any training
logic; it only launches training.py normally, once per seed. Only the
requested SEED changes between runs -- all other hyperparameters
(TOTAL_EPISODES, entropy coef, etc.) stay whatever training.py's current
settings are.

WARNING: each run is a full training run (e.g. 4000 episodes on the line
map), so this will take roughly N_SEEDS times as long as one normal
training.py run. Expect this to take a while -- consider starting it and
checking back later, or running overnight.

After all seeds finish, run summarize_multiseed.py to collect the results
into the mean/variance table Bhaskar asked for.

Usage:
    python run_multiseed.py [seed1 seed2 seed3 ...]
    (defaults to seeds 42 43 44 45 46 if none given)
"""

import sys
import os
import subprocess
import time

SEEDS = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [42, 43, 44, 45, 46]


def main():
    print(f'Running training.py for {len(SEEDS)} seeds: {SEEDS}')
    t0 = time.time()
    for i, seed in enumerate(SEEDS):
        tag = f'_seed{seed}'
        print(f'\n{"=" * 70}\n[{i+1}/{len(SEEDS)}] Starting run for SEED={seed} '
              f'(output prefix: mappo_line{tag})\n{"=" * 70}')
        env = os.environ.copy()
        env['MAPPO_TRAIN_SEED'] = str(seed)
        env['MAPPO_RUN_TAG'] = tag
        run_t0 = time.time()
        result = subprocess.run([sys.executable, 'training.py'], env=env)
        run_duration = time.time() - run_t0
        status = 'OK' if result.returncode == 0 else f'FAILED (exit code {result.returncode})'
        print(f'[{i+1}/{len(SEEDS)}] SEED={seed} finished in '
              f'{run_duration / 60:.1f} min -- {status}')
        if result.returncode != 0:
            print(f'  Stopping early due to failure. Fix the error and re-run '
                  f'with the remaining seeds: python run_multiseed.py {" ".join(str(s) for s in SEEDS[i:])}')
            return

    total = time.time() - t0
    print(f'\nAll {len(SEEDS)} seeds complete in {total / 60:.1f} min total.')
    print('Now run: python summarize_multiseed.py ' + ' '.join(str(s) for s in SEEDS))


if __name__ == '__main__':
    main()

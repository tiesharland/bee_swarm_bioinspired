import numpy as np
import pandas as pd
from scipy.stats import qmc
import random
from classes import *  # assumes your Environment and Bee live here
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ---------------------------
# Simulator wrapper
# ---------------------------
def run(inpt, vis=False, max_steps=False, seed=None):
    env = Environment(
        inpt['width'], inpt['length'], inpt['hive_radius'], inpt['nectar_count'],
        inpt['max_nec_strength'], inpt['idle_prob'], inpt['follow_prob'],
        max_st=True, hive_pos='centre', seed=seed
    )
    num_scouts = int(inpt['num_bees'] * inpt['perc_scouts'])
    for i in range(inpt['num_bees']):
        sc = i < num_scouts
        b = Bee(env, inpt['sense_range'], inpt['dt'], inpt['kappa_0'], inpt['alpha'],
                inpt['beta'], inpt['w_dir'], scout=sc)
        env.add_bee(b)

    total = sum([nec['strength'] for nec in env.nectars])
    t = 0
    time_first_nect = None
    look_first_nect = True

    while len(env.nectars) > 0:
        env.update()
        t += 1
        if env.dances and look_first_nect:
            time_first_nect = t
            look_first_nect = False
        if max_steps and t >= inpt['max_steps']:
            break

    success = len(env.nectars) == 0
    time = t if success else None

    if vis:
        env.visualise()

    return {
        'time_to_depletion': time,
        'time_to_first_nectar': time_first_nect,
        'success': success
    }

# ---------------------------
# Experiment setup
# ---------------------------
base_config = {'width': 10, 'length': 10, 'hive_radius': 0.2, 'nectar_count': 10, 'max_nec_strength': 5,
               'sense_range': 0.5, 'dt': 0.2, 'num_bees': 20, 'max_steps': 20000,
                'idle_prob': None, 'follow_prob': None, 'perc_scouts': None,
                'kappa_0': None, 'alpha': None, 'beta': None, 'w_dir': None}

param_bounds = {'idle_prob': (0.0, 0.5), 'follow_prob': (0.4, 0.99), 'perc_scouts': (0.05, 0.95),
                'kappa_0': (0.1, 5.0), 'alpha': (0.1, 50.0), # log-scale
                'beta': (0.5, 10.0), # log-scale
                'w_dir': (0.1, 0.9)}

# ---------------------------
# LHS sampling
# ---------------------------
def latin_hypercube_samples(n_samples, param_bounds):
    sampler = qmc.LatinHypercube(d=len(param_bounds))
    sample = sampler.random(n=n_samples)
    l_bounds = [param_bounds[p][0] for p in param_bounds]
    u_bounds = [param_bounds[p][1] for p in param_bounds]
    scaled = qmc.scale(sample, l_bounds, u_bounds)
    return [dict(zip(param_bounds.keys(), row)) for row in scaled]

# ---------------------------
# Worker for parallel execution
# ---------------------------
def run_single(config, sample_id, rep):
    seed = random.randint(0, 1_000_000)
    result = run(config, vis=False, seed=seed, max_steps=config['max_steps'])
    return {
        'sample_id': sample_id,
        'rep': rep,
        **{k: config[k] for k in param_bounds.keys()},
        'time_to_depletion': result['time_to_depletion'],
        'time_to_first_nectar': result['time_to_first_nectar'],
        'success': result['success']
    }

# ---------------------------
# Run experiment with multiprocessing + tqdm
# ---------------------------
def run_experiment(n_samples=20, n_reps=5, outfile="results.csv", n_workers=None, diagnostic=False):
    if diagnostic:
        print("Running diagnostic mode: small test run")
        n_samples = min(n_samples, 5)
        n_reps = min(n_reps, 2)
        outfile = "bee_results_diagnostic.csv"

    configs = latin_hypercube_samples(n_samples, param_bounds)
    tasks = []

    # Build all tasks
    for i, pset in enumerate(configs):
        cfg = dict(base_config)
        cfg.update(pset)
        for rep in range(n_reps):
            tasks.append((cfg, i, rep))

    records = []
    total_runs = len(tasks)

    # Default: use all available cores
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_single, cfg, i, rep) for (cfg, i, rep) in tasks]
        for f in tqdm(as_completed(futures), total=total_runs, desc="Running simulations"):
            records.append(f.result())

    df = pd.DataFrame(records)

    # Print basic summary
    print("\nDiagnostic summary:" if diagnostic else "\nExperiment summary:")
    print(df[['time_to_depletion', 'time_to_first_nectar', 'success']].describe())

    # Compute and print total success rate
    total_success_rate = df['success'].mean()
    print(f"\nTotal success rate across all runs: {total_success_rate:.2%}")

    df.to_csv(outfile, index=False)
    print(f"Saved results to {outfile}")
    return df

# ---------------------------
# Run if main
# ---------------------------
if __name__ == "__main__":
    # Example: use only half your cores
    df_diag = run_experiment(diagnostic=True)
    df = run_experiment(n_samples=200, n_reps=15, outfile="bee_results.csv", n_workers=4)
    print(df.head())

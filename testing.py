import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from classes import *
from run import run

# === Simulation wrapper with seed control ===
def run_simulation(params: dict, seed: int) -> dict:
    np.random.seed(seed)
    return run(params, vis=False, max_steps=False)


# === Experiment grid runner ===
def grid_experiment(base_params, p_name, q_name, p_vals, q_vals, R=5, n_jobs=-1):
    """
    Run a grid of experiments varying p_name and q_name.
    Each (p, q) pair is replicated R times with different seeds.
    """
    tasks = []
    for p, q in itertools.product(p_vals, q_vals):
        for r in range(R):
            params = base_params.copy()
            params[p_name] = p
            params[q_name] = q
            tasks.append((params, p, q, r))

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one)(params, p, q, r, p_name, q_name)
        for (params, p, q, r) in tqdm(tasks, desc="Running simulations")
    )

    df = pd.DataFrame(results)
    return df


def run_one(params, p, q, r, p_name, q_name):
    res = run_simulation(params, seed=r)
    res.update({ p_name: p, q_name: q, "rep": r })
    return res


# === Summarizing ===
def summarize_grid(df, p_name, q_name, metric='time_to_depletion'):
    pivot_mean = df.pivot_table(values=metric, index=q_name, columns=p_name, aggfunc='mean')
    pivot_std = df.pivot_table(values=metric, index=q_name, columns=p_name, aggfunc='std')
    return pivot_mean, pivot_std


# === Visualization ===
def plot_heatmap(pivot, p_vals, q_vals, p_name, q_name, title=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, xticklabels=np.round(p_vals, 2), yticklabels=np.round(q_vals, 2),
                cmap="viridis", annot=True, fmt=".1f")
    plt.xlabel(p_name)
    plt.ylabel(q_name)
    if title:
        plt.title(title)
    plt.show()


# === Example run ===
if __name__ == "__main__":
    base_params = {
        'width': 10,
        'length': 10,
        'hive_radius': 0.2,
        'nectar_count': 15,
        'max_nec_strength': 5,
        'num_bees': 10,
        'dt': 0.2,
        'max_steps': 5000,
        # behavioural
        'idle_prob': 0.2,
        'follow_prob': 0.5,
        'perc_scouts': 0.3,
        'sense_range': 0.5,
        # movement
        'kappa_0': 10,
        'alpha': 10,
        'beta': 20,
        'w_dir': 0.5
    }

    # Choose which two parameters to vary
    p_name = 'idle_prob'
    q_name = 'follow_prob'
    p_vals = np.linspace(0.05, 0.5, 5)
    q_vals = np.linspace(0.1, 0.9, 5)

    # Run experiments
    df = grid_experiment(base_params, p_name, q_name, p_vals, q_vals, R=10, n_jobs=4)
    print("Finished simulations!")
    print(df.head())

    # Summarize
    pivot_mean, pivot_std = summarize_grid(df, p_name, q_name, metric='time_to_depletion')

    # Plot
    plot_heatmap(pivot_mean, p_vals, q_vals, p_name, q_name,
                 title="Mean time to depletion")


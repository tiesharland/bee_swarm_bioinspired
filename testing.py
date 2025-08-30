import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count

from classes import *
from run import run  # your modified run() with fixed hive

# ==== DEFAULT PARAMETERS ====
default_params = {
    'width': 10,
    'length': 10,
    'hive_radius': 0.2,
    'nectar_count': 15,
    'max_nec_strength': 5,
    'num_bees': 10,
    'dt': 0.2,
    'max_steps': 50000,
    'idle_prob': 0.2,
    'follow_prob': 0.5,
    'perc_scouts': 0.3,
    'sense_range': 0.5,  # fixed
    'kappa_0': 10,
    'alpha': 10,
    'beta': 20,
    'w_dir': 0.5,
}

# ==== PARAMETER RANGES (sense_range removed) ====
param_ranges = {
    'idle_prob':    np.linspace(0.0, 0.5, 5),
    'follow_prob':  np.linspace(0.0, 1.0, 5),
    'perc_scouts':  np.linspace(0.1, 0.9, 5),
    'kappa_0':      [1, 5, 10, 20, 50],
    'alpha':        [1, 5, 10, 20, 50],
    'beta':         [1, 5, 10, 20, 50],
    'w_dir':        np.linspace(0.0, 1.0, 5),
}

params_of_interest = list(param_ranges.keys())
pairs = list(itertools.combinations(params_of_interest, 2))

# ==== SINGLE SIMULATION ====
def run_single_sim(params, p_name, q_name, p_val, q_val, rep):
    params_copy = params.copy()
    params_copy[p_name] = float(p_val)
    params_copy[q_name] = float(q_val)
    out = run(params_copy, vis=False, max_steps=True, seed=63)

    # Fill ALL parameters explicitly
    for k, v in params.items():
        if k not in out:
            out[k] = v

    # Overwrite varied ones
    out[p_name] = p_val
    out[q_name] = q_val
    out["rep"] = rep
    return out

# Wrapper for multiprocessing
def run_single_sim_wrapper(args):
    return run_single_sim(*args)

# ==== PARALLEL GRID RUN ====
def run_grid_parallel(p_name, q_name, n_reps=5):
    results = []
    tasks = [(default_params, p_name, q_name, p_val, q_val, rep)
             for p_val in param_ranges[p_name]
             for q_val in param_ranges[q_name]
             for rep in range(n_reps)]

    with Pool(processes=cpu_count()) as pool:
        for out in tqdm(pool.imap_unordered(run_single_sim_wrapper, tasks),
                        total=len(tasks),
                        desc=f"Sweeping {p_name} vs {q_name}",
                        ncols=100):
            results.append(out)

    return pd.DataFrame(results)

# ==== SUMMARIZE RESULTS ====
def summarize_grid(df, p_name, q_name):
    p_vals = param_ranges[p_name]
    q_vals = param_ranges[q_name]

    success_pivot = df.pivot_table(
        values="success",
        index=q_name,
        columns=p_name,
        aggfunc="mean"
    ).reindex(index=q_vals, columns=p_vals, fill_value=0)

    df_success = df[df["success"] == True]

    mean_time = df_success.pivot_table(
        values="time_to_depletion",
        index=q_name,
        columns=p_name,
        aggfunc="mean"
    ).reindex(index=q_vals, columns=p_vals, fill_value=0)

    std_time = df_success.pivot_table(
        values="time_to_depletion",
        index=q_name,
        columns=p_name,
        aggfunc="std"
    ).reindex(index=q_vals, columns=p_vals, fill_value=0)

    return success_pivot, mean_time, std_time

# ==== PLOTTING ====
def create_heatmap(pivot, p_name, q_name, title, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, ax=ax)
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=45)
    ax.set_yticklabels([f"{y:.2f}" for y in pivot.index], rotation=0)
    ax.set_title(title)
    ax.set_xlabel(p_name)
    ax.set_ylabel(q_name)
    return fig

# ==== MAIN LOOP ====
if __name__ == "__main__":
    all_results = []
    all_figures = []

    for p_name, q_name in tqdm(pairs, desc="All parameter pairs", ncols=100):
        df = run_grid_parallel(p_name, q_name, n_reps=5)
        all_results.append(df)

        success, mean_time, std_time = summarize_grid(df, p_name, q_name)

        # all_figures.append(create_heatmap(success, p_name, q_name, f"Success rate: {p_name} vs {q_name}"))
        all_figures.append(create_heatmap(mean_time, p_name, q_name, f"Mean time-to-depletion: {p_name} vs {q_name}"))
        all_figures.append(create_heatmap(std_time, p_name, q_name, f"Std time-to-depletion: {p_name} vs {q_name}"))

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("pairwise_sensitivity_results.csv", index=False)
    print("Saved results to pairwise_sensitivity_results.csv")

    plt.show()


import pandas as pd
import numpy as np

# --------------------------------------------------
# Load pairwise results
# --------------------------------------------------
df = pd.read_csv("pairwise_sensitivity_results.csv")

# Parameters to analyze
param_cols = ['idle_prob', 'follow_prob', 'perc_scouts',
              'kappa_0', 'alpha', 'beta', 'w_dir']

# --------------------------------------------------
# Step 1: Compute efficiency
# --------------------------------------------------
df['efficiency'] = df.apply(
    lambda row: row['total_nectar_collected'] / row['time_to_depletion']
    if row['time_to_depletion'] > 0 else 0.0, axis=1
)

# --------------------------------------------------
# Step 2: Aggregate over repetitions
# --------------------------------------------------
agg_df = df.groupby(param_cols).agg(
    time_to_depletion_mean=('time_to_depletion', 'mean'),
    time_to_depletion_std=('time_to_depletion', 'std'),
    efficiency_mean=('efficiency', 'mean'),
    efficiency_std=('efficiency', 'std')
).reset_index()

# --------------------------------------------------
# Step 3: Compute parameter importance
# --------------------------------------------------
importance_time = {}
importance_eff = {}

for p in param_cols:
    grouped_time = agg_df.groupby(p)['time_to_depletion_mean'].mean()
    importance_time[p] = grouped_time.max() - grouped_time.min()

    grouped_eff = agg_df.groupby(p)['efficiency_mean'].mean()
    importance_eff[p] = grouped_eff.max() - grouped_eff.min()

importance_time = dict(sorted(importance_time.items(), key=lambda x: x[1], reverse=True))
importance_eff = dict(sorted(importance_eff.items(), key=lambda x: x[1], reverse=True))

# --------------------------------------------------
# Step 4: Estimate optimal values
# --------------------------------------------------
optimal_time = {}
optimal_eff = {}

for p in param_cols:
    grouped_time = agg_df.groupby(p)['time_to_depletion_mean'].mean()
    optimal_time[p] = grouped_time.idxmin()  # fastest depletion

    grouped_eff = agg_df.groupby(p)['efficiency_mean'].mean()
    optimal_eff[p] = grouped_eff.idxmax()  # highest efficiency

# --------------------------------------------------
# Step 5: Display summary
# --------------------------------------------------
print("\n=== Parameter importance (time_to_depletion) ===")
for p, val in importance_time.items():
    print(f"{p}: {val:.2f}")

print("\n=== Parameter importance (efficiency) ===")
for p, val in importance_eff.items():
    print(f"{p}: {val:.4f}")

print("\n=== Estimated optimal values (fastest depletion) ===")
for p, val in optimal_time.items():
    print(f"{p}: {val}")

print("\n=== Estimated optimal values (highest efficiency) ===")
for p, val in optimal_eff.items():
    print(f"{p}: {val}")

# --------------------------------------------------
# Step 6: Save summary to CSV
# --------------------------------------------------
summary_df = pd.DataFrame({
    'parameter': param_cols,
    'importance_time': [importance_time[p] for p in param_cols],
    'importance_efficiency': [importance_eff[p] for p in param_cols],
    'optimal_time': [optimal_time[p] for p in param_cols],
    'optimal_efficiency': [optimal_eff[p] for p in param_cols]
})

summary_df.to_csv("pairwise_sensitivity_summary.csv", index=False)
print("\nSaved summary -> pairwise_sensitivity_summary.csv")

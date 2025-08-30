import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Load results
# ---------------------------
df = pd.read_csv("bee_results.csv")

# ---------------------------
# Count failed replicates
# ---------------------------
num_failed = df['time_to_depletion'].isna().sum()
total_runs = len(df)
print(f"Total runs: {total_runs}")
print(f"Failed runs (NaN time_to_depletion): {num_failed}")
print(f"Successful runs: {total_runs - num_failed}\n")

# ---------------------------
# Keep only the replicate with the quickest time_to_depletion per sample_id safely
# ---------------------------
agg_list = []

for sample_id, group in df.groupby("sample_id"):
    # Drop NaNs in this group
    group_valid = group.dropna(subset=["time_to_depletion"])
    if len(group_valid) == 0:
        # skip parameter sets that failed all replicates
        continue
    # pick the row with minimum time_to_depletion
    min_idx = group_valid["time_to_depletion"].idxmin()
    agg_list.append(df.loc[min_idx])

agg = pd.DataFrame(agg_list).reset_index(drop=True)

print("Aggregated results (quickest time_to_depletion per parameter set):")
print(agg.head())

# ---------------------------
# Scatterplots: each parameter vs quickest time_to_depletion
# ---------------------------
params = ["idle_prob", "follow_prob", "perc_scouts", "kappa_0", "alpha", "beta", "w_dir"]

plt.figure(figsize=(16, 10))
for i, p in enumerate(params, 1):
    plt.subplot(2, 4, i)
    plt.scatter(agg[p], agg["time_to_depletion"], alpha=0.7)
    plt.xlabel(p)
    plt.ylabel("time_to_depletion" if i in [1,5] else "")
    plt.title(f"{p} vs time_to_depletion")
plt.tight_layout()
plt.show()

# ---------------------------
# Pairplot showing performance (time_to_depletion) as hue
# ---------------------------
plot_vars = params  # keep only the parameters on axes
hue_var = "time_to_depletion"  # continuous performance metric

sns.pairplot(
    agg,
    vars=plot_vars,
    hue=hue_var,
    palette="viridis",
    diag_kind="kde",
    corner=True,
    plot_kws={"alpha": 0.8, "s": 40}  # adjust marker transparency and size
)
plt.suptitle("Pairplot of parameters colored by time_to_depletion", y=1.02)
plt.show()

# ---------------------------
# Correlation heatmap
# ---------------------------
corr = agg[params + ["time_to_depletion"]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation heatmap (quickest time_to_depletion)")
plt.show()


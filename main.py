# import pandas as pd
# from testing import default_params, param_ranges, params_of_interest
#
#
# df = pd.read_csv("pairwise_sensitivity_results.csv")
# subset_defaults = {k: default_params[k] for k in params_of_interest}
#
# # Fill NaNs with default parameter values
# for k, v in subset_defaults.items():
#     df[k] = df[k].fillna(v)
#
# df.to_csv("pairwise_sensitivity_results.csv", index=False)
# Drop rows where time_to_depletion is NaN (unsuccessful runs)
# df_valid = df.dropna(subset=["time_to_depletion"])
#
# # Find the row with the minimum completion time
# fastest_row = df_valid.loc[df_valid["time_to_depletion"].idxmin()]
#
# print("Fastest run:")
# print(fastest_row)

import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("pairwise_sensitivity_results.csv")

# Only successful runs
df_valid = df[df["success"] == True].copy()

# Group by parameter set
param_cols = [c for c in df_valid.columns if c not in ["rep", "success", "time_to_depletion"]]
stats_df = df_valid.groupby(param_cols)["time_to_depletion"].agg(['mean', 'std']).reset_index()

# Replace NaN std with 0
stats_df['std'] = stats_df['std'].fillna(0)

# Top 5 fastest average depletion
top5 = stats_df.sort_values('mean').head(5)

print("Top 5 parameter sets (fastest average depletion) with std:")
print(top5)

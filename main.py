import numpy as np
from run import run


num_runs = 100
range_follow_prob = np.arange(0.5, 0.8, 10)
range_idle_prob = np.arange(0.1, 0.5, 10)
range_num_bees = 20
range_num_nectar = 10

for f in range_follow_prob:
    for i in range_idle_prob:


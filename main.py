import numpy as np
from run import run


num_runs = 100
range_follow_prob = np.arange(0.5, 0.8, 10)
range_idle_prob = np.arange(0.1, 0.5, 10)
range_num_bees = 20
range_num_nectar = 10

for f in range_follow_prob:
    for i in range_idle_prob:
        for b in range_num_bees:
            for n in range_num_nectar:
                inp = dict()
                inp['width'] = 4
                inp['length'] = 4
                inp['hive_radius'] = 0.2
                inp['max_nec_strength'] = 50
                inp['idle_prob'] = i
                inp['follow_prob'] = f
                inp['nectar_count'] = n
                inp['num_bees'] = b
                inp['sense_range'] = .5
                inp['dt'] = 0.1
                inp['n_time_steps'] = 1000
                inp['max_steps'] = 5000

                run(inp)


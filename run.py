from classes import *


def run(inpt, vis=True, max_steps=False, seed=None):
    env = Environment(inpt['width'], inpt['length'], inpt['hive_radius'], inpt['nectar_count'],
                      inpt['max_nec_strength'], inpt['idle_prob'], inpt['follow_prob'],
                      max_st=True, hive_pos='centre', seed=seed)
    num_scouts = int(inpt['num_bees'] * inpt['perc_scouts'])
    for i in range(inpt['num_bees']):
        sc = i < num_scouts
        b = Bee(env, inpt['sense_range'], inpt['dt'], inpt['kappa_0'], inpt['alpha'],
                inpt['beta'], inpt['w_dir'], scout=sc)
        env.add_bee(b)

    # print(f'------------------------------------------------------')
    #     # f'\n------------------------------------------------------')
    total = sum([nec['strength'] for nec in env.nectars])
    t = 0
    time_first_nect = None
    look_first_nect = True
    while len(env.nectars) > 0:
        env.update()
        t += 1
        if env.dances and look_first_nect:
            # print(f'Time to find first nectar: {t} steps')
            time_first_nect = t
            look_first_nect = False
        if max_steps and t >= inpt['max_steps']:
            break

    # Determine success
    success = len(env.nectars) == 0
    time = t if success else None

    if vis:
        env.visualise()

    return {
        'time_to_depletion': time,
        'total_nectar_collected': total,
        'time_to_first_nectar': time_first_nect,
        'success': success}



if __name__ == '__main__':
    inp = {
        'width': 10,
        'length': 10,
        'hive_radius': 0.2,
        'nectar_count': 15,
        'max_nec_strength': 5,
        'num_bees': 10,
        'dt': 0.2,
        'max_steps': 10000,
        'sense_range': 0.5,
        # behavioural defaults:
        'idle_prob': 0.25,
        'follow_prob': 1.0,
        'perc_scouts': 0.9,
        # movement params (for kappa etc)
        'kappa_0': 1.0,
        'alpha': 50.0,
        'beta': 5,
        'w_dir': 1.0
    }

    results = run(inp, vis=True, max_steps=True, seed=63)
    print(results)

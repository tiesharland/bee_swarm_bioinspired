from classes import *


def run(inpt, vis=True, max_steps=True):
    env = Environment(inpt['width'], inpt['length'], inpt['hive_radius'], inpt['nectar_count'],
                      inpt['max_nec_strength'], inpt['idle_prob'], inpt['follow_prob'])
    num_scouts = inpt['num_bees'] * inpt['perc_scouts']
    for i in range(inpt['num_bees']):
        sc = False
        if i < num_scouts:
            sc = True
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
        if t > inpt['max_steps'] and max_steps:
            break
        if env.dances and look_first_nect:
            # print(f'Time to find first nectar: {t} steps')
            time_first_nect = t
            look_first_nect = False

    if len(env.nectars) > 0:
        # print(f'Not all nectar sources depleted in {inpt['max_steps']} time steps.'
        #       f'\nThere are {len(env.nectars)} nectar sources remaining.')
        success = False
        time = None
    else:
        # elif len(env.nectars) == 0:
        # print(f'No nectar sources remaining.')
        # print(f'Iterations needed for finding all nectars: {t} time steps')
        success = True
        time = t

    # print(f'Number of recorded frames: {len(env.history)}')
    if vis:
        env.visualise()

    return {'time_to_depletion': time, 'total_nectar_collected': total,
            'time_to_first_nectar': time_first_nect, 'success': success}



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
        # behavioural defaults:
        'idle_prob': 0.2,
        'follow_prob': 0.5,
        'perc_scouts': 0.3,
        'sense_range': 0.5,
        # movement params (for kappa etc)
        'kappa_0': 10,
        'alpha': 10,
        'beta': 20,
        'w_dir': 0.5
    }

    results = run(inp)
    print(results)

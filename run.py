from classes import *


def run(inpt, vis=True):
    env = Environment(inpt['width'], inpt['length'], inpt['hive_radius'], inpt['nectar_count'],
                      inpt['max_nec_strength'], inpt['idle_prob'], inpt['follow_prob'])
    num_scouts = inpt['num_bees'] * inpt['perc_scouts']
    for i in range(inpt['num_bees']):
        sc = False
        if i < num_scouts:
            sc = True
        b = Bee(env, inpt['sense_range'], inpt['dt'], inpt['kappa_min'], inpt['alpha'], inpt['beta'],
                inpt['w_dir'], inpt['w_rep'], scout=sc)
        env.add_bee(b)

    print(f'------------------------------------------------------')
        # f'\n------------------------------------------------------')

    t = 0
    look_first_nect = True
    while len(env.nectars) > 0 and t <= inpt['max_steps']:
        env.update()
        t += 1
        if env.dances and look_first_nect:
            print(f'Time to find first nectar: {t} steps')
            look_first_nect = False

    if len(env.nectars) > 0:
        print(f'Not all nectar sources depleted in {inpt['max_steps']} time steps.'
              f'\nThere are {len(env.nectars)} nectar sources remaining.')
    else:
        print(f'Iterations needed for finding all nectars: {t} time steps')

    # print(f'Number of recorded frames: {len(env.history)}')
    if vis:
        env.visualise()


if __name__ == '__main__':
    inp = dict()
    inp['width'] = 10
    inp['length'] = 10
    inp['hive_radius'] = 0.2
    inp['max_nec_strength'] = 5
    inp['idle_prob'] = 0.2
    inp['follow_prob'] = 0.5
    inp['nectar_count'] = 15
    inp['num_bees'] = 20
    inp['perc_scouts'] = .3
    inp['sense_range'] = .5
    inp['kappa_min'] = 10
    inp['alpha'] = 10
    inp['beta'] = 20
    inp['w_dir'] = 0.5
    inp['w_rep'] = 0.5
    inp['dt'] = 0.2
    inp['max_steps'] = 10000

    run(inp)

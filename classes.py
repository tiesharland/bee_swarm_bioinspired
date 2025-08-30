import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation


class Environment:
    def __init__(self, width, length, hive_radius, nectar_count, max_nec_strength, idle_prob, follow_prob,
                 max_st, hive_pos, seed=None):
        self.width = width
        self.length = length
        self.hive_radius = hive_radius
        self.max_nec_strength = max_nec_strength
        self.idle_prob = idle_prob
        self.follow_prob = follow_prob
        self.nectars = self.place_nectar(nectar_count, max_st, seed)
        self.hive_position = self.place_hive(hive_pos)
        self.bees = []
        self.dances = []
        self.history = []

    def place_nectar(self, num, max_st, seed):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        nectars = []
        for n in range(num):
            x = np.random.uniform(0, self.length)
            y = np.random.uniform(0, self.width)
            stren = (self.max_nec_strength if max_st else np.random.randint(1, self.max_nec_strength + 1))
            nectars.append({'position': (x, y), 'strength': stren})
        return nectars

    def place_hive(self, hive_pos):
        if hive_pos == 'centre':
            return self.length/2, self.width/2
        elif hive_pos == 'random':
            return np.random.uniform(0, self.length), np.random.uniform(0, self.width)
        else:
            ValueError(f'Not valid hive position: {hive_pos} should be "centre" or "random"')

    def add_bee(self, bee):
        self.bees.append(bee)

    def add_dance(self, direction, distance, strength):
        self.dances.append({'direction': direction, 'distance': distance, 'strength': strength})

    def update(self):
        for b in self.bees:
            b.update()
        self.nectars = [n for n in self.nectars if n['strength'] > 0]
        self.record_state()

    def record_state(self):
        bee_data = [{'position': bee.position, 'state': bee.state} for bee in self.bees]
        nectar_data = [{'position': nectar['position'], 'strength': nectar['strength']} for nectar in self.nectars]
        self.history.append({'bee_data': bee_data, 'nectar_data': nectar_data})

    def visualise(self, fps=30, filename=None):
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.width)

        title = ax.set_title(f'Bee swarm foraging')

        bee_scat = ax.scatter([], [], color='black', s=50)
        nectar_scat = ax.scatter([], [], color='orange', marker='*', s=100, alpha=0.0)
        hive_circle = patches.Circle(self.hive_position, radius=self.hive_radius,
                                     facecolor='gold', edgecolor='black', label='Hive')
        ax.add_patch(hive_circle)

        def init():
            bee_scat.set_offsets(np.empty((0, 2)))
            nectar_scat.set_offsets(np.empty((0, 2)))
            nectar_scat.set_alpha(0)
            return [bee_scat, nectar_scat, hive_circle, title]

        def update(frame):
            bee_positions = [b['position'] for b in self.history[frame]['bee_data']]
            bee_states = [b['state'] for b in self.history[frame]['bee_data']]
            bee_scat.set_offsets(np.array(bee_positions))

            nectar_positions = []
            nectar_alphas = []
            for nec in self.history[frame]['nectar_data']:
                nectar_positions.append(nec['position'])
                strength = nec['strength']
                nectar_alphas.append(np.clip(strength / self.max_nec_strength, 0, 1))

            if nectar_positions:
                nectar_scat.set_offsets(np.array(nectar_positions))
                nectar_scat.set_alpha(nectar_alphas)
            else:
                # safely reset to empty without touching alpha
                nectar_scat.set_offsets(np.empty((0, 2)))

            bees_in_hive = sum(np.linalg.norm(np.array(pos) - np.array(self.hive_position)) <= self.hive_radius
                               for pos in bee_positions)

            # Update the title
            title.set_text(f"Bee swarm foraging - Bees in hive: {bees_in_hive}")

            return [bee_scat, nectar_scat, title]

        ani = animation.FuncAnimation(fig, update, init_func=init, frames=len(self.history), interval=1000/fps, blit=False)

        if filename:
            ani.save(filename)

        plt.show()

    def plot_grid(self, step):
        fig, ax = plt.subplots(1, 1)
        ax.set_title(f'Bee swarm - step {step}' if step is not None else 'Bee swarm')
        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.width)

        for nec in self.nectars:
            x, y = nec['position']
            s = nec['strength']
            alph = s / self.max_nec_strength
            ax.scatter(x, y, color='orange', s=100, alpha=alph, marker='*', label='Nectar')

        hive_circle = patches.Circle(self.hive_position, radius=self.hive_radius,
                                     facecolor='gold', edgecolor='black', label='Hive')
        ax.add_patch(hive_circle)

        for bee in self.bees:
            if bee.state not in ['home', 'dancing']:
                bx, by = bee.position
                ax.scatter(bx, by, color='black', s=50)
        plt.show()


class Bee:
    def __init__(self, environment, sense_range, dt, kappa_0, alpha, beta, w_dir, scout=False):
        self.env = environment
        self.sense_range = sense_range
        self.dt = dt
        self.kappa_0 = kappa_0
        self.alpha = alpha
        self.beta = beta
        self.w_dir = w_dir
        self.w_rep = 1 - w_dir
        self.scout = scout
        self.position = self.env.hive_position
        self.state = "home"
        self.dance = 0
        self.found_nectar = []
        self.known_nectars = []
        self.path_history = [self.position]
        self.target = None

    def sense_nectar(self):
        new_nectar = []
        for nec in self.env.nectars:
            dist = np.linalg.norm(np.array(nec['position']) - np.array(self.position))
            if dist <= self.sense_range:
                if nec not in self.known_nectars:
                    self.known_nectars.append(nec)
                    new_nectar.append(nec)
        self.found_nectar = new_nectar

    def move(self, random=False):
        if random:
            if len(self.path_history) >= 2:
                last_x, last_y = self.path_history[-1]
                x, y = self.position
                direction_vec = np.array([x - last_x, y - last_y])
            else:
                direction_vec = np.array(self.position) - np.array(self.env.hive_position)
            if np.linalg.norm(direction_vec) > 0:
                direction_vec = direction_vec / np.linalg.norm(direction_vec)
            else:
                direction_vec = np.zeros(2)
            repulsion_vec = np.zeros(2)
            margin = self.sense_range
            if self.position[0] < margin:
                repulsion_vec[0] += 1
            elif self.position[0] > self.env.length - margin:
                repulsion_vec[0] -= 1
            if self.position[1] < margin:
                repulsion_vec[1] += 1
            elif self.position[1] > self.env.width - margin:
                repulsion_vec[1] -= 1
            if np.linalg.norm(repulsion_vec) > 0:
                repulsion_vec = repulsion_vec / np.linalg.norm(repulsion_vec)
            combined_vec = self.w_dir * direction_vec + self.w_rep * repulsion_vec
            if np.linalg.norm(combined_vec) > 0:
                combined_vec = combined_vec / np.linalg.norm(combined_vec)
                pref_angle = np.arctan2(combined_vec[1], combined_vec[0])
            else:
                pref_angle = np.random.uniform(0, 2 * np.pi)
            distance_from_hive = np.linalg.norm(np.array(self.position) - np.array(self.env.hive_position))
            kappa = self.kappa_0 + self.alpha * np.exp(-distance_from_hive / self.beta)
            angle = np.random.vonmises(mu=pref_angle, kappa=kappa)
            dx, dy = self.dt * np.cos(angle), self.dt * np.sin(angle)
        elif self.target:
            dx, dy = self.dt * self.target["direction"][0], self.dt * self.target["direction"][1]
        else:
            raise ValueError("Not random (searching) and no target")
        x = np.clip(self.position[0] + dx, 0, self.env.width)
        y = np.clip(self.position[1] + dy, 0, self.env.length)
        # dx_real, dy_real = x - self.position[0], y - self.position[1]
        self.position = (x, y)
        self.path_history.append(self.position)

    def update(self):
        if self.state == "following":
            if self.target:
                dist_to_hive = np.linalg.norm(np.array(self.position) - np.array(self.env.hive_position))
                self.sense_nectar()
                if self.found_nectar:
                    self.state = "found"
                    self.target = None
                elif dist_to_hive >= self.target["distance"] - self.sense_range:
                    self.state = ("searching" if self.scout else "returning")
                    self.target = None
                else:
                    self.move()
            elif self.env.dances:
                self.target = np.random.choice(self.env.dances)
                self.sense_nectar()
                if self.found_nectar:
                    self.state = "found"
                    self.target = None
                else:
                    self.move()
            else:
                self.state = ("searching" if self.scout else "returning")
                self.target = None
        elif self.state == "searching":
            self.sense_nectar()
            dist_to_hive = np.linalg.norm(np.array(self.position) - np.array(self.env.hive_position))
            if self.found_nectar:
                self.state = "found"
            elif dist_to_hive <= self.env.hive_radius:
                if self.env.dances:
                    self.state = "following"
                    self.target = np.random.choice(self.env.dances)
                else:
                    self.move(random=True)
            else:
                self.move(random=True)
        elif self.state == "home":
            if self.known_nectars:
                nec = np.random.choice(self.known_nectars)
                vector = np.array(nec['position']) - np.array(self.position)
                distance = np.linalg.norm(vector)
                direction = tuple(vector / distance)
                self.known_nectars.clear()
                exists = any(np.allclose(d["direction"], direction) and np.isclose(d["distance"], distance)
                             for d in self.env.dances)
                if not exists:
                    strength = nec["strength"]
                    self.env.add_dance(direction, distance, strength)
                    self.state = "dancing"
                    self.target = {'direction': direction, 'distance': distance, 'strength': strength}
                    self.dance = 1
            elif self.target:
                self.state = "following"
            else:
                if self.scout and np.random.rand() > self.env.idle_prob:
                    self.state = "searching"
                else:
                    if self.env.dances and np.random.rand() < self.env.follow_prob:
                        self.state = "following"
                        self.target = np.random.choice(self.env.dances)

                # self.state = np.random.choice(["home", "following", "searching"],
                #                                   p=[self.env.idle_prob, self.env.follow_prob,
                #                                      1 - self.env.idle_prob - self.env.follow_prob])
                # if self.state == "searching":
                #     self.target = None
                # elif self.state == "following":
                #     if not self.target:
                #         if self.env.dances:
                #             self.target = np.random.choice(self.env.dances)
                #         else:
                #             self.state = "searching"
        elif self.state == "found":
            nec = None
            for n in self.found_nectar:
                vector = np.array(n['position']) - np.array(self.position)
                distance = np.linalg.norm(vector)
                direction = vector / distance
                if self.target and np.allclose(direction, self.target["direction"]):
                    nec = n
                    break
            if nec is None:
                nec = np.random.choice(self.found_nectar)
            nec['strength'] -= 1
            dist_to_hive = np.linalg.norm(np.array(self.position) - np.array(self.env.hive_position))
            if dist_to_hive <= self.env.hive_radius:
                self.state = "home"
                self.position = self.env.hive_position
                self.path_history.clear()
            else:
                self.state = "returning"
        elif self.state == "returning":
            # if self.path_history:
            #     self.position = (self.path_history[-1][0], self.path_history[-1][1])
            #     self.path_history.pop()
            vec_to_home = np.array(self.env.hive_position) - np.array(self.position)
            dist_to_hive = np.linalg.norm(vec_to_home)
            if dist_to_hive <= self.env.hive_radius:
                self.state = "home"
                self.position = self.env.hive_position
                self.path_history.clear()
            else:
                vec_to_home = tuple(vec_to_home / dist_to_hive)
                self.position = (self.position[0] + self.dt * vec_to_home[0], self.position[1] + self.dt * vec_to_home[1])
                self.path_history.append(self.position)
        elif self.state == "dancing":
            if self.dance > self.target['strength']:
                if self.env.dances:
                    for dance in self.env.dances:
                        if (np.allclose(dance["direction"], self.target["direction"]) and
                                np.isclose(dance["distance"], self.target["distance"])):
                            self.env.dances.remove(dance)
                            # self.target = None
                self.state = "home"
                self.dance = 0
                # if self.target:
                #     for dance in self.env.dances:
                #         if (np.allclose(dance["direction"], self.target["direction"]) and
                #                 np.isclose(dance["distance"], self.target["distance"])):
                #             self.env.dances.remove(dance)
            else:
                self.dance += 1


if __name__ == "__main__":
    width = 4
    length = 4
    hive_radius = 0.2
    max_nec_strength = 4
    idle_prob = 0.1
    follow_prob = 0.8
    nectar_count = 3
    num_bees = 4
    sense_range = .5
    dt = 0.1
    n_time_steps = 1000
    max_steps = 5000
    kappa_0 = 10
    alpha = 10
    beta = 20
    w_dir = 0.5
    filename = 'swarm_test'

    env = Environment(width, length, hive_radius, nectar_count, max_nec_strength, idle_prob, follow_prob)
    for i in range(num_bees):
        b = Bee(env, sense_range, dt, kappa_0, alpha, beta, w_dir)
        env.add_bee(b)

    t = 0
    # env.plot_grid(t)
    while len(env.nectars) > 0 and t <= max_steps:
        env.update()
        # print(f'------------------------------------------------------')
        # print(f'Time step: {t}\n------------------------------------------------------')
        # print(f'Dances: {env.dances}')
        # for id, b in enumerate(env.bees):
        #     print(f'Bee {id}: {b.state}')
        #     print(f'       Target: {b.target}')
        #     if b.state == "found":
        #         print(f'       Position: {b.position}')
        #     # elif b.state == "following":
        #     #     print(f'       Target: {b.target}')
        # env.plot_grid(t)
        t += 1

    print(
        f'------------------------------------------------------'
        f'\n------------------------------------------------------')

    if len(env.nectars) > 0:
        print(f'Not all nectars found in {max_steps} time steps.\nThere are {len(env.nectars)} nectars remaining.')
    else:
        print(f'Iterations needed for finding all nectars: {t} time steps')

    print(f'Number of recorded frames: {len(env.history)}')
    env.visualise()

    # for t in range(1, n_time_steps + 1):
    #     env.update()
    #     print(f'------------------------------------------------------')
    #     print(f'Time step: {t}\n------------------------------------------------------')
    #     print(f'Dances: {env.dances}')
    #     for id, b in enumerate(env.bees):
    #         print(f'Bee {id}: {b.state}')
    #         print(f'       Target: {b.target}')
    #         if b.state == "found":
    #             print(f'       Position: {b.position}')
    #         # elif b.state == "following":
    #         #     print(f'       Target: {b.target}')
    #     env.visualise(t)

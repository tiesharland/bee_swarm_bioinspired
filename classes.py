import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Environment:
    def __init__(self, width, length, hive_radius, nectar_count, max_nec_strength):
        self.width = width
        self.length = length
        self.hive_radius = hive_radius
        self.max_nec_strength = max_nec_strength
        self.nectars = self.place_nectar(nectar_count)
        self.hive_position = self.place_hive()
        self.bees = []
        self.dances = []
        self.idle_prob = 0.1
        self.follow_prob = 0.8

    def place_nectar(self, num):
        nectars = []
        for n in range(num):
            x = np.random.uniform(0, self.length)
            y = np.random.uniform(0, self.width)
            nectars.append({'position': (x, y), 'strength': np.random.randint(1, self.max_nec_strength + 1)})
        return nectars

    def place_hive(self):
        return np.random.uniform(0, self.length), np.random.uniform(0, self.width)

    def add_bee(self, bee):
        self.bees.append(bee)

    def add_dance(self, direction, distance):
        self.dances.append({'direction': direction, 'distance': distance})

    def update(self):
        for b in self.bees:
            b.update()

    def visualise(self, step):
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
    def __init__(self, environment, sense_range, dt):
        self.env = environment
        self.sense_range = sense_range
        self.dt = dt
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
            combined_vec = 0.5 * direction_vec + 0.5 * repulsion_vec
            if np.linalg.norm(combined_vec) > 0:
                combined_vec = combined_vec / np.linalg.norm(combined_vec)
                pref_angle = np.arctan2(combined_vec[1], combined_vec[0])
            else:
                pref_angle = np.random.uniform(0, 2 * np.pi)
            distance_from_hive = np.linalg.norm(np.array(self.position) - np.array(self.env.hive_position))
            kappa = 10 + 10 * np.exp(-distance_from_hive / 20)
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
                    self.state = "searching"
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
                self.state = "searching"
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
                    self.env.add_dance(direction, distance)
                    self.state = "dancing"
                    self.target = {'direction': direction, 'distance': distance}
                    self.dance = 1
            elif not self.env.dances:
                if np.random.rand() > self.env.idle_prob:
                    self.state = "searching"
                    self.target = None
            else:
                self.state = np.random.choice(["idle", "following", "searching"],
                                              p=[self.env.idle_prob, self.env.follow_prob,
                                                 1 - self.env.idle_prob - self.env.follow_prob])
                if self.state == "searching":
                    self.target = None
                elif self.state == "following":
                    if not self.target:
                        self.target = np.random.choice(self.env.dances)
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
            if nec['strength'] <= 0:
                for n in self.env.nectars:
                    if np.allclose(n["position"], nec["position"]):
                        self.env.nectars.remove(n)
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
            vec_to_home = tuple(vec_to_home / dist_to_hive)
            if dist_to_hive <= self.env.hive_radius:
                self.state = "home"
                self.position = self.env.hive_position
                self.path_history.clear()
            else:
                self.position = (self.position[0] + self.dt * vec_to_home[0], self.position[1] + self.dt * vec_to_home[1])
                self.path_history.append(self.position)
        elif self.state == "dancing":
            if self.dance > 4:
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
    nectar_count = 3
    num_bees = 4
    sense_range = .5
    dt = 0.1
    n_time_steps = 1000

    env = Environment(width, length, hive_radius, max_nec_strength, nectar_count)
    for i in range(num_bees):
        b = Bee(env, sense_range, dt)
        env.add_bee(b)

    env.visualise(0)

    for t in range(1, n_time_steps + 1):
        env.update()
        print(f'------------------------------------------------------')
        print(f'Time step: {t}\n------------------------------------------------------')
        print(f'Dances: {env.dances}')
        for id, b in enumerate(env.bees):
            print(f'Bee {id}: {b.state}')
            print(f'       Target: {b.target}')
            if b.state == "found":
                print(f'       Position: {b.position}')
            # elif b.state == "following":
            #     print(f'       Target: {b.target}')
        env.visualise(t)

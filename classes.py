import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, width, length, nectar_count):
        self.width = width
        self.length = length
        self.nectar_count = nectar_count
        self.nectar_positions = self.place_nectar()
        self.hive_position = self.place_hive()
        self.bees = []
        self.dances = []
        self.probabilities = [0.1, 0.9]             # Exploration, Following waggle

    def place_nectar(self):
        positions = []
        for n in range(self.nectar_count):
            x = np.random.uniform(0, self.length)
            y = np.random.uniform(0, self.width)
            positions.append((x, y))
        return positions

    def place_hive(self):
        return np.random.uniform(0, self.length), np.random.uniform(0, self.width)

    def add_bee(self, bee):
        self.bees.append(bee)

    def add_dance(self, direction, distance):
        if {'direction': direction, 'distance': distance} not in self.dances:
            self.dances.append({'direction': direction, 'distance': distance})

    def update(self):
        for b in self.bees:
            b.update()

    def visualise(self, step):
        fig, ax = plt.subplots(1, 1)
        ax.set_title(f'Bee swarm - step {step}' if step is not None else 'Bee swarm')
        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.width)

        for x, y in self.nectar_positions:
            ax.scatter(x, y, color='orange', s=100, marker='*', label='Nectar')

        hx, hy = self.hive_position
        ax.scatter(hx, hy, c='gold', s=200, edgecolors='black', label='Hive')

        for bee in self.bees:
            if bee.state != "home":
                bx, by = bee.position
                ax.scatter(bx, by, color='black', s=80)
        fig.show()


class Bee:
    def __init__(self, environment, sense_range, dt):
        self.env = environment
        self.sense_range = sense_range
        self.dt = dt
        self.position = env.hive_position
        self.state = "searching"
        self.found_nectar = False
        self.known_nectar_pos = []
        self.path_history = [(0, 0)]
        self.target = None

    def sense_nectar(self):
        px, py = self.position
        self.known_nectar_pos = []
        for nx, ny in self.env.nectar_positions:
            dist = np.sqrt((px - nx) ** 2 + (py - ny) ** 2)
            if dist <= self.sense_range:
                self.known_nectar_pos.append((nx, ny))
        self.found_nectar = len(self.known_nectar_pos) > 0
        
    def move_random(self):
        angle = np.random.uniform(0, 2 * np.pi)
        dx, dy = self.dt * np.cos(angle), self.dt * np.sin(angle)
        x = np.clip(self.position[0] + dx, 0, self.env.width)
        y = np.clip(self.position[1] + dy, 0, self.env.length)
        self.position = (x, y)
        self.path_history.append((dx, dy))
        
    def update(self):
        if self.state == "following":
            if self.target:
                self.sense_nectar()
                if self.found_nectar:
                    self.state = "found"
                else:
                    dx, dy = self.dt * self.target["direction"][0], self.dt * self.target["direction"][1]
                    self.position = (self.position[0] + dx, self.position[1] + dy)
            elif self.env.dances:
                self.target = np.random.choice(self.env.dances)
                self.env.dances.remove(self.target)
                self.sense_nectar()
                if self.found_nectar:
                    self.state = "found"
                else:
                    dx, dy = self.dt * self.target["direction"][0], self.dt * self.target["direction"][1]
                    self.position = (self.position[0] + dx, self.position[1] + dy)
            else:
                self.state = "searching"
        elif self.state == "searching":
            self.sense_nectar()
            if self.found_nectar:
                self.state = "found"
            else:
                self.move_random()
        elif self.state == "home":
            if self.known_nectar_pos:
                nec = self.known_nectar_pos[np.random.choice(len(self.known_nectar_pos))]
                dx, dy = nec[0] - self.position[0], nec[1] - self.position[1]
                distance = np.sqrt(dx ** 2 + dy ** 2)
                direction = (dx / distance, dy / distance)
                self.env.add_dance(direction, distance)
                self.known_nectar_pos.clear()
                self.state = "dancing"
            else:
                self.state = np.random.choice(["searching", "following"], p=self.env.probabilities)
        elif self.state == "found":
            if self.position == self.env.hive_position:
                self.state = "dancing"
            else:
                self.state = "returning"
        elif self.state == "returning":
            self.position = (self.position[0] - self.dt * self.path_history[-1][0],
                             self.position[1] - self.dt * self.path_history[-1][1])
            self.path_history.pop()
            dist = np.sqrt((self.position[0] - self.env.hive_position[0]) ** 2
                            + (self.position[1] - self.env.hive_position[1]) ** 2)
            if dist <= self.sense_range:
                self.state = "home"
                self.position = self.env.hive_position


if __name__ == "__main__":
    dt = 0.1
    n_time_steps = 100
    S = 1

    env = Environment(4, 4, 3)
    for i in range(2):
        b = Bee(env, S, dt)
        env.add_bee(b)

    env.visualise(0)

    for t in range(1, n_time_steps + 1):
        env.update()
        print(f'------------------------------------------------------')
        print(f'Time step: {t}\n------------------------------------------------------')
        print(f'Dances: {env.dances}')
        for id, b in enumerate(env.bees):
            print(f'Bee {id}: {b.state}')
            if b.state == "found":
                print(f'       Path: {b.path_history}\n       Position: {b.position}')
            elif b.state == "following":
                print(b.target)
        env.visualise(t)

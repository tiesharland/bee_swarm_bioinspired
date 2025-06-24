import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, width, length, nectar_count):
        self.width = width
        self.length = length
        self.nectar_count = nectar_count
        self.grid = np.zeros((self.width, self.length), dtype=int)
        self.nectar_positions = self.place_nectar()
        self.hive_position = self.place_hive()
        self.probabilities = [0.1, 0.9]             # Exploration, Following waggle
        self.bees = []
        self.dances = []

    def place_nectar(self):
        empty = list(zip(*np.where(self.grid == 0)))
        nectar_positions = [empty[i] for i in np.random.choice(len(empty), self.nectar_count, replace=False)]
        positions = []
        for x, y in nectar_positions:
            self.grid[x, y] = 1
            positions.append((x, y))
        return np.array(positions)

    def place_hive(self):
        while True:
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.length)
            if self.grid[x, y] == 0:
                self.grid[x, y] = 2
                return x, y

    def add_bee(self, bee):
        self.bees.append(bee)

    def add_dance(self, direction):
        if {'direction': direction} not in self.dances:
            self.dances.append({'direction': direction})

    def update(self):
        self.grid[self.grid == 3] = 0
        for b in self.bees:
            b.update()
            # if b.state == "found":
            #     self.found_nectars += b.known_nectar_pos
            if b.position != self.hive_position:
                self.grid[b.position[0], b.position[1]] = 3


class Bee:
    def __init__(self, environment, sense_range):
        self.env = environment
        self.sense_range = sense_range
        self.position = env.hive_position
        self.state = "searching"
        self.found_nectar = False
        self.known_nectar_pos = []
        self.path_history = [(0, 0)]
        self.target = None

    def sense_nectar(self):
        r = self.sense_range
        self.known_nectar_pos = []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                x, y = self.position[0] + i, self.position[1] + j
                if 0 <= x < self.env.width and 0 <= y < self.env.length:
                    if self.env.grid[x, y] == 1:
                        self.known_nectar_pos.append((x, y))
                        self.env.grid[x, y] = 4                 # Found nectar
        self.found_nectar = len(self.known_nectar_pos) > 0

    def update(self):
        if self.state == "home":
            if self.known_nectar_pos:
                nec = self.known_nectar_pos[np.random.choice(len(self.known_nectar_pos))]
                direction = (nec[0] - self.position[0], nec[1] - self.position[1])
                self.env.add_dance(direction)
                self.known_nectar_pos.clear()
                self.state = "dancing"
            else:
                self.state = np.random.choice(["searching", "following"], p=self.env.probabilities)
        elif self.state == "searching":
            self.sense_nectar()
            if self.found_nectar:
                self.state = "found"
            else:
                random_moves = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                                if not ((dx == 0 and dy == 0) or (-dx, -dy) == self.path_history[-1])]
                dx, dy = random_moves[np.random.choice(len(random_moves))]
                x = max(0, min(self.env.width - 1, self.position[0] + dx))
                y = max(0, min(self.env.length - 1, self.position[1] + dy))
                dx, dy = x - self.position[0], y - self.position[1]
                self.position = (x, y)
                self.path_history.append((dx, dy))
        elif self.state == "following":
            self.target = np.random.choice(self.env.dances)['direction']
        elif self.state == "found":
            if self.position == self.env.hive_position:
                self.state = "home"
            else:
                self.state = "returning"
        elif self.state == "returning":
            self.position = self.position[0] - self.path_history[-1][0], self.position[1] - self.path_history[-1][1]
            self.path_history.pop()
            if self.position == self.env.hive_position:
                self.state = "home"


if __name__ == "__main__":
    env = Environment(8, 8, 3)
    for i in range(2):
        b = Bee(env, 1)
        env.add_bee(b)

    dt = 0.1
    n_time_steps = 10
    for t in range(n_time_steps):
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

    print(env.grid)

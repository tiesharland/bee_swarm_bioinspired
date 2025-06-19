import numpy as np
import matplotlib.pyplot as plt


class Environment():
    def __init__(self, width, length, nectar_count):
        self.width = width
        self.length = length
        self.nectar_count = nectar_count
        self.grid = np.zeros((self.width, self.length), dtype=int)
        self.nectar_positions = self.place_nectar()
        self.hive_position = self.place_hive()

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


class Bee:
    def __init__(self, env):
        self.env = env
        self.sense_range = 2
        self.position = env.hive_position
        self.state = "searching"
        self.found_nectar = False
        self.known_nectar_pos = None

    def sense_nectar(self):
        pass


if __name__ == "__main__":
    env = Environment(2, 2, 2)
    print(env.grid)
    print(env.nectar_positions)
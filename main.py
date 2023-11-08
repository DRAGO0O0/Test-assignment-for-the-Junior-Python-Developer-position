import random
import heapq
import numpy as np
import matplotlib.pyplot as plt

class CityGrid:
    def __init__(self, rows, columns, num_towers, filled_space=0.3, ):
        self.rows = rows
        self.columns = columns
        self.grid = self.create_grid(filled_space)
        self.num_towers = num_towers
        self.adjacency_list = [[] for _ in range(num_towers)]


    def create_grid(self, filled_space):
        grid = [[False] * self.columns for _ in range(self.rows)]
        square_filled = int(filled_space * self.rows * self.columns)

        i_j = [(i, j) for i in range(self.rows) for j in range(self.columns)]
        random.shuffle(i_j)

        for i in range(square_filled):
            row, col = i_j[i]
            grid[row][col] = True

        return grid

    def is_square_filled(self, row, column):
        return self.grid[row][column]


    def place_tower(self, row, column, range_R):
        for i in range(max(0, row - range_R), min(row + range_R + 1, self.rows)):
            for j in range(max(0, column - range_R), min(column + range_R +1, self.columns)):
                self.grid[i][j] = True

    def visualize_tower_coverage(self):
        for row in self.grid:
            for tower in row:
                if tower:
                    print("x")
                else:
                    print(".")
            print()

    def minimize_place_tower(self, range_R):
        covered_blocks = set()

        for row in range(self.rows):
            for col in range(self.columns):
                if not self.is_square_filled(row, col) and (row, col) not in covered_blocks:
                    self.place_tower(row, col, range_R)
                    for i in range(max(0, row - range_R), min(row + range_R + 1, self.rows)):
                        for j in range(max(0, col - range_R), min(col + range_R + 1, self.columns)):
                            covered_blocks.add((i, j))

    def visualize_towers(self):
        for row in self.grid:
            for tower in row:
                if tower:
                    print("x")
                else:
                    print(".")
            print()

    def add_link(self, tower1, tower2, reliability):
        self.adjacency_list[tower1].append(tower2, reliability)
        self.adjacency_list[tower2].append(tower1, reliability)

    def find_most_reliable_path(self, source, destination):
        pq = []
        dist = [float('-inf')] * self.num_towers
        prev = [None] * self.num_towers


        dist[source] = 1
        heapq.heappush(pq, (-1, source))

        while pq:
            current_reliability, current_tower = heapq.heappop(pq)

            if current_tower == destination:
                break

            for neighbor, link_reliability in self.adjacency_list[current_tower]:
                new_reliability = dist[current_tower] * link_reliability

                if new_reliability > dist[neighbor]:
                    dist[neighbor] = new_reliability
                    prev[neighbor] = current_tower
                    heapq.heappush(pq, (-new_reliability, neighbor))

        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()

        return path

    def visualize_city_grid(city):
        grid = np.array(city.grid)
        plt.imshow(grid, cmap='gray', interpolation='nearest')
        plt.title('City Grid')
        plt.show()

    def visualize_towers(city):
        grid = np.array(city.grid)
        tower_locations = np.argwhere(grid)
        plt.scatter(tower_locations[:, 1], tower_locations[:, 0], marker='x', color='red')
        plt.imshow(grid, cmap='gray', interpolation='nearest')
        plt.title('Towers')
        plt.show()

    def visualize_coverage_areas(city, tower_range):
        grid = np.array(city.grid)
        coverage_grid = np.zeros_like(grid, dtype=int)

        tower_indices = np.argwhere(grid)
        for tower_index in tower_indices:
            row, col = tower_index
            coverage_grid[max(0, row - tower_range): min(row + tower_range + 1, city.rows),
            max(0, col - tower_range): min(col + tower_range + 1, city.columns)] = 1

        plt.imshow(coverage_grid, cmap='gray', interpolation='nearest')
        plt.title('Coverage Areas')
        plt.show()

    def visualize_data_path(city, network, source, destination):
        grid = np.array(city.grid)
        path = network.find_most_reliable_path(source, destination)
        path_grid = np.zeros_like(grid, dtype=int)

        path_indices = np.unravel_index(path, grid.shape)
        path_grid[path_indices] = 1

        plt.imshow(path_grid, cmap='gray', interpolation='nearest')
        plt.title('Data Path')
        plt.show()




















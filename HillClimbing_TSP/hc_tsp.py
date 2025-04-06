import numpy as np
import matplotlib.pyplot as plt
import time
import os
import imageio

class TSPEngine:
    def __init__(self, cities):
        self.cities = np.array(cities)
        self.num_cities = len(cities)
        self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        dist = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                d = np.linalg.norm(self.cities[i] - self.cities[j])
                dist[i][j] = dist[j][i] = d
        return dist

    def tour_length(self, path):
        cost = 0
        for i in range(len(path)):
            cost += self.distance_matrix[path[i - 1]][path[i]]
        return cost

    def get_neighbors(self, path):
        neighbors = []
        for i in range(1, self.num_cities - 1):
            for j in range(i + 1, self.num_cities):
                new_path = path[:]
                new_path[i], new_path[j] = new_path[j], new_path[i]
                neighbors.append(new_path)
        return neighbors

    def plot_tour(self, path, title="TSP Tour"):
        path = np.array(path)
        tour_coords = self.cities[path]
        plt.figure(figsize=(8, 5))
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'o-', label='Path')
        plt.plot([tour_coords[-1, 0], tour_coords[0, 0]],
                 [tour_coords[-1, 1], tour_coords[0, 1]], 'o--', color='gray')
        for i, (x, y) in enumerate(self.cities):
            plt.text(x, y, str(i), fontsize=12)
        plt.title(title)
        plt.grid()
        plt.show()

def generate_random_cities(n=10, seed=42):
    np.random.seed(seed)
    return np.random.rand(n, 2) * 100

if __name__ == "__main__":
    cities = generate_random_cities(10)
    env = TSPEngine(cities)
    
    initial_path = list(range(len(cities)))
    np.random.shuffle(initial_path)

    print("Initial tour cost:", env.tour_length(initial_path))
    env.plot_tour(initial_path, "Initial Random Tour")

def hill_climbing(env, max_iterations=1000, verbose=True):
    current_path = list(range(env.num_cities))
    np.random.shuffle(current_path)
    current_cost = env.tour_length(current_path)

    if verbose:
        print("Initial cost:", current_cost)

    start_time = time.time()
    iteration = 0

    while iteration < max_iterations:
        neighbors = env.get_neighbors(current_path)
        neighbor_costs = [env.tour_length(p) for p in neighbors]

        best_neighbor_idx = np.argmin(neighbor_costs)
        best_neighbor = neighbors[best_neighbor_idx]
        best_cost = neighbor_costs[best_neighbor_idx]

        if best_cost < current_cost:
            current_path = best_neighbor
            current_cost = best_cost
            if verbose:
                print(f"Iteration {iteration + 1}: cost = {current_cost:.2f}")
        else:
            if verbose:
                print("No better neighbor found. Reached local optimum.")
            break

        iteration += 1

    end_time = time.time()
    time_taken = end_time - start_time

    reward = -current_cost
    convergence_point = iteration

    print("\n--- Hill Climbing Stats ---")
    print(f"Reward: {reward:.2f}")
    print(f"Time: {time_taken:.4f} seconds")
    print(f"Point of Convergence: {convergence_point} iterations")

    return current_path, current_cost, convergence_point, time_taken

def save_tour_frame(cities, path, filename, title):
    plt.figure(figsize=(6, 4))
    coords = cities[path]
    plt.plot(coords[:, 0], coords[:, 1], 'o-', color='blue')
    plt.plot([coords[-1, 0], coords[0, 0]], [coords[-1, 1], coords[0, 1]], 'o--', color='gray')
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(path[i]), fontsize=9)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def hill_climbing_with_gif(env, gif_path="tsp_hc.gif", max_iterations=1000):
    current_path = list(range(env.num_cities))
    np.random.shuffle(current_path)
    current_cost = env.tour_length(current_path)

    frames_dir = "hc_frames"
    os.makedirs(frames_dir, exist_ok=True)

    save_tour_frame(env.cities, current_path, f"{frames_dir}/frame_000.png", f"Start: {current_cost:.2f}")
    frame_files = [f"{frames_dir}/frame_000.png"]

    iteration = 0
    while iteration < max_iterations:
        neighbors = env.get_neighbors(current_path)
        neighbor_costs = [env.tour_length(p) for p in neighbors]

        best_neighbor_idx = np.argmin(neighbor_costs)
        best_neighbor = neighbors[best_neighbor_idx]
        best_cost = neighbor_costs[best_neighbor_idx]

        if best_cost < current_cost:
            current_path = best_neighbor
            current_cost = best_cost
            filename = f"{frames_dir}/frame_{iteration+1:03d}.png"
            save_tour_frame(env.cities, current_path, filename, f"Iter {iteration+1}: {current_cost:.2f}")
            frame_files.append(filename)
        else:
            break
        iteration += 1

    with imageio.get_writer(gif_path, mode='I', duration=0.6) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)

    print(f"GIF saved at: {gif_path}")
    return current_path, current_cost, iteration

def evaluate_hill_climbing(env_generator, num_runs=5, plot_path="hc_avg_time.png"):
    times = []
    iterations = []

    for i in range(num_runs):
        env = env_generator()
        _, _, steps, time_taken = hill_climbing(env, verbose=False)
        times.append(time_taken)
        iterations.append(steps)
        print(f"Run {i+1}: Time = {time_taken:.4f}s, Convergence at step = {steps}")

    avg_time = sum(times) / len(times)
    avg_steps = sum(iterations) / len(iterations)

    print(f"\nAverage Time: {avg_time:.4f} seconds")
    print(f"Average Convergence Steps: {avg_steps:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_runs + 1), times, marker='o', label='Time per run')
    plt.axhline(avg_time, color='red', linestyle='--', label=f'Avg Time = {avg_time:.4f}s')
    plt.title("Hill Climbing: Time to Reach Optimum")
    plt.xlabel("Run")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved average time plot to: {plot_path}")

def env_generator():
    cities = generate_random_cities(10)
    return TSPEngine(cities)

evaluate_hill_climbing(env_generator, num_runs=5, plot_path="hc_avg_time.png")

if __name__ == "__main__":
    cities = generate_random_cities(10)
    env = TSPEngine(cities)
    best_path, best_cost, steps = hill_climbing_with_gif(env, gif_path="hill_climbing_tsp.gif")

    print(f"Final Cost: {best_cost:.2f} in {steps} steps")
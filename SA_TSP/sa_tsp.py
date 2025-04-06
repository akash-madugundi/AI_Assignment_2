import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------- Utilities ----------
def generate_cities(n_cities=20, seed=42):
    np.random.seed(seed)
    return np.random.rand(n_cities, 2) * 100

def euclidean_distance(city1, city2):
    return np.linalg.norm(city1 - city2)

def create_distance_matrix(cities):
    n = len(cities)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = euclidean_distance(cities[i], cities[j])
    return matrix

def compute_total_distance(tour, dist_matrix):
    return sum(dist_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def plot_frame(ax, cities, tour, title):
    ax.clear()
    path = cities[tour + [tour[0]]]
    ax.plot(path[:, 0], path[:, 1], 'o-', color='blue')
    for i, (x, y) in enumerate(cities):
        ax.text(x + 1, y + 1, str(i), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

# ---------- Simulated Annealing ----------
def simulated_annealing_tsp(dist_matrix, cities=None, gif_path=None,
                             initial_temp=1000, cooling_rate=0.995,
                             stopping_temp=1e-3, max_iter=100000, capture_interval=500):
    n = len(dist_matrix)
    current = list(range(n))
    random.shuffle(current)
    best = current[:]
    best_cost = compute_total_distance(best, dist_matrix)
    current_cost = best_cost
    T = initial_temp
    iteration = 0
    convergence_iter = 0
    tour_history = [best[:]] if gif_path and cities is not None else []

    while T > stopping_temp and iteration < max_iter:
        i, j = random.sample(range(n), 2)
        neighbor = current[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbor_cost = compute_total_distance(neighbor, dist_matrix)

        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best = current
                best_cost = current_cost
                convergence_iter = iteration

        if gif_path and iteration % capture_interval == 0:
            tour_history.append(best[:])

        T *= cooling_rate
        iteration += 1

    if gif_path and cities is not None:
        fig, ax = plt.subplots(figsize=(6, 5))

        def update(frame):
            plot_frame(ax, cities, tour_history[frame], f"Step {frame * capture_interval}")

        anim = FuncAnimation(fig, update, frames=len(tour_history), interval=200)
        anim.save(gif_path, writer=PillowWriter(fps=5))
        print(f"GIF saved to: {gif_path}")

    return best, best_cost, convergence_iter

# ---------- Test with Multiple Runs (Single City Map) ----------
def test_tsp_algorithm_multiple_runs(n_cities=20, seed=2024, num_runs=5):
    times = []
    rewards = []
    convergences = []

    print("Generating city map once...\n")
    cities = generate_cities(n_cities, seed)
    dist_matrix = create_distance_matrix(cities)

    for run in range(num_runs):
        print(f"Run {run+1}:")
        start_time = time.time()

        if run == 0:
            gif_path = "sa_tsp.gif"
            tour, best_cost, convergence_iter = simulated_annealing_tsp(dist_matrix, cities, gif_path=gif_path)
        else:
            tour, best_cost, convergence_iter = simulated_annealing_tsp(dist_matrix)

        end_time = time.time()
        exec_time = end_time - start_time
        reward = -best_cost

        times.append(exec_time)
        rewards.append(reward)
        convergences.append(convergence_iter)

        print(f"  Time Taken: {exec_time:.4f}s | Reward: {reward:.2f} | Convergence Iteration: {convergence_iter}")

    avg_time = np.mean(times)
    avg_reward = np.mean(rewards)
    avg_conv = int(np.mean(convergences))

    print("\n=== Averages Across Runs ===")
    print(f"Average Time: {avg_time:.4f} seconds")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Convergence Iteration: {avg_conv}")

    # Save PNG of Execution Time Plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, num_runs + 1), times, color='skyblue')
    plt.axhline(avg_time, color='red', linestyle='--', label=f"Avg: {avg_time:.4f}s")
    plt.xlabel("Run")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time per Run (Simulated Annealing on TSP)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sa_avg_time.png", dpi=300)
    print("Saved average time plot as sa_avg_time.png")

# ---------- Main ----------
if __name__ == "__main__":
    test_tsp_algorithm_multiple_runs(n_cities=20, seed=2024, num_runs=5)
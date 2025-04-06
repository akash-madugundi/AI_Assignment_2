import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio
import os

def create_frozen_lake_env(custom_desc=None, map_name="4x4", is_slippery=False, render_mode="ansi"):
    if custom_desc:
        env = gym.make(
            "FrozenLake-v1",
            desc=custom_desc,
            is_slippery=is_slippery,
            render_mode=render_mode
        )
    else:
        env = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode=render_mode
        )
    return env

def get_pos_from_index(index, size):
    return divmod(index, size)

def get_index_from_pos(pos, size):
    return pos[0] * size + pos[1]

def manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def valid_moves(pos, grid):
    """Return valid next positions from current pos."""
    moves = []
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    rows, cols = len(grid), len(grid[0])

    for dr, dc in directions:
        r, c = pos[0] + dr, pos[1] + dc
        if 0 <= r < rows and 0 <= c < cols:
            if grid[r][c] != 'H':
                moves.append((r, c))
    return moves

def branch_and_bound_frozen_lake(env, timeout=600):
    start_time = time.time()
    desc = env.unwrapped.desc.astype(str)
    size = desc.shape[0]

    start_pos = None
    goal_pos = None

    for r in range(size):
        for c in range(size):
            if desc[r][c] == 'S':
                start_pos = (r, c)
            elif desc[r][c] == 'G':
                goal_pos = (r, c)

    pq = []
    visited = set()
    heapq.heappush(pq, (manhattan(start_pos, goal_pos), 0, start_pos, []))
    best_path = None
    best_cost = float('inf')
    reward = 0
    convergence_point = start_pos

    while pq and time.time() - start_time < timeout:
        f, cost, pos, path = heapq.heappop(pq)
        if pos in visited:
            continue
        visited.add(pos)
        convergence_point = pos

        if pos == goal_pos:
            best_path = path
            best_cost = cost
            reward = 1
            print("**Goal reached!**")
            break

        for move in valid_moves(pos, desc):
            if move not in visited:
                new_cost = cost + 1
                h = manhattan(move, goal_pos)
                heapq.heappush(pq, (new_cost + h, new_cost, move, path + [move]))

    total_time = time.time() - start_time

    print("\nSearch finished.")
    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Nodes explored: {len(visited)}")
    if best_path:
        print(f"Best path (length {len(best_path)}): {best_path}")
    else:
        print("No solution found within time limit.")

    print(f"Point of convergence: {convergence_point}")
    print(f"Final reward: {reward}")

    return best_path, best_cost, reward, convergence_point


def visualize_path(env, path):
    desc = env.unwrapped.desc.astype(str)
    vis = desc.copy()
    for r, c in path:
        if vis[r][c] == 'F':
            vis[r][c] = '.'
    print("\n📍 Path visualization:")
    for row in vis:
        print("".join(row))

def animate_path(path, desc, filename="bnb_run.gif", interval=0.5):
    size = len(desc)
    frames = []
    tile_colors = {
        'S': 'lightgreen',
        'G': 'gold',
        'H': 'black',
        'F': 'white',
        '.': 'skyblue'
    }

    for step in range(1, len(path) + 1):
        fig, ax = plt.subplots(figsize=(size, size))
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

        for r in range(size):
            for c in range(size):
                tile = desc[r][c]
                color = tile_colors.get(tile, 'white')
                rect = patches.Rectangle((c, size - r - 1), 1, 1, facecolor=color, edgecolor='gray')
                ax.add_patch(rect)

        for (r, c) in path[:step]:
            rect = patches.Rectangle((c, size - r - 1), 1, 1, facecolor='skyblue', edgecolor='blue')
            ax.add_patch(rect)

        temp_path = f"_frame_{step}.png"
        plt.savefig(temp_path)
        plt.close()
        frames.append(imageio.imread(temp_path))
        os.remove(temp_path)

    imageio.mimsave(filename, frames, duration=interval)
    print(f"GIF saved to: {filename}")

def run_multiple_tests(runs=5):
    times = []
    rewards = []
    convergence_points = []
    maps_used = []

    os.makedirs("gifs", exist_ok=True)

    print("\n=== Running Multiple Tests with Random Maps ===\n")

    for i in range(runs):
        print(f"Run {i + 1}/{runs}")

        random_map = generate_random_map(size=8)
        maps_used.append(random_map)
        env = create_frozen_lake_env(custom_desc=random_map, is_slippery=False)

        print("Random Map Used:")
        for row in random_map:
            print(row)

        start = time.time()
        path, cost, reward, convergence = branch_and_bound_frozen_lake(env)
        end = time.time()
        elapsed = end - start

        times.append(elapsed)
        rewards.append(reward)
        convergence_points.append(convergence)

        print(f"Reward: {reward}, Time: {elapsed:.2f}s, Point of Convergence: {convergence}")

        if path:
            gif_path = f"gifs/bnb_run_{i + 1}.gif"
            animate_path(path, np.array(random_map), filename=gif_path, interval=0.3)

        env.close()
        print("")

    print("\nSummary:")
    print("Run\tReward\tTime (s)\tConvergence Point")
    for i in range(runs):
        print(f"{i+1}\t{rewards[i]}\t{times[i]:.2f}\t\t{convergence_points[i]}")

    avg_time = np.mean(times)
    success_rate = sum(rewards) / runs * 100
    print(f"\nAverage Time: {avg_time:.2f} sec")
    print(f"Success Rate: {success_rate:.0f}%")

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, runs + 1), times, color='skyblue')
    plt.axhline(avg_time, color='red', linestyle='--', label=f'Avg Time = {avg_time:.2f}s')
    plt.xlabel("Run")
    plt.ylabel("Time (s)")
    plt.title("BnB Search Time Across Random Maps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bnb_avg_time_plot.png")
    plt.show()

if __name__ == "__main__":
    run_multiple_tests(runs=5)
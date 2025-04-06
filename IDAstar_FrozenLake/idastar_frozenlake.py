import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os
import time

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_pos_from_index(index, width):
    return divmod(index, width)

def ida_star(env):
    start_state, _ = env.reset()
    desc = env.unwrapped.desc
    goal_state = np.where(np.array(desc, dtype="U1") == "G")
    goal_pos = (goal_state[0][0], goal_state[1][0])
    width = desc.shape[1]

    def search(path, g, threshold):
        state = path[-1]
        pos = get_pos_from_index(state, width)
        h = manhattan_distance(pos, goal_pos)
        f = g + h

        if f > threshold:
            return f
        if state == goal_pos[0] * width + goal_pos[1]:
            return "FOUND"

        min_threshold = float("inf")
        for action in range(env.action_space.n):
            transitions = env.unwrapped.P[state][action]
            for prob, next_state, reward, terminated in transitions:
                if prob == 0:
                    continue
                if next_state in path:
                    continue
                path.append(next_state)
                t = search(path, g + 1, threshold)
                if t == "FOUND":
                    return "FOUND"
                if t < min_threshold:
                    min_threshold = t
                path.pop()
        return min_threshold

    threshold = manhattan_distance(get_pos_from_index(start_state, width), goal_pos)
    path = [start_state]
    while True:
        t = search(path, 0, threshold)
        if t == "FOUND":
            return path
        if t == float("inf"):
            return None
        threshold = t

def run_ida_star_on_frozenlake(desc=None):
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
    start_time = time.time()

    path = ida_star(env)

    total_time = time.time() - start_time
    reward = 0
    point_of_convergence = 0

    if path:
        print("\nPath found by IDA* (length = {}):".format(len(path)))
        width = env.unwrapped.desc.shape[1]
        for idx in path:
            pos = divmod(idx, width)
            print(f" -> {pos}", end="")
        print()

        state, _ = env.reset()
        for step in path[1:]:
            for action in range(env.action_space.n):
                transitions = env.unwrapped.P[state][action]
                for prob, next_state, r, terminated in transitions:
                    if prob > 0 and next_state == step:
                        reward = r
                        state = next_state
                        break
                else:
                    continue
                break

        point_of_convergence = len(path)
        print(f"\nReward: {reward}")
        print(f"Time Taken: {total_time:.4f} seconds")
        print(f"Point of Convergence (steps): {point_of_convergence}")

        visualize_path(path, env.unwrapped.desc)

    else:
        print("No path found.")
        print(f"Time Taken: {total_time:.4f} seconds")
        print(f"Point of Convergence: N/A")
        print(f"Reward: 0")

def visualize_path(path, desc, gif_name="ida_star_path.gif", failure=False):
    desc = np.array([list(row) for row in desc], dtype="U1")
    n_rows, n_cols = desc.shape
    frames = []

    for step_idx in range(max(1, len(path))):
        fig, ax = plt.subplots(figsize=(6, 6))

        for i in range(n_rows):
            for j in range(n_cols):
                tile = desc[i, j]
                color = {
                    'S': 'green',
                    'G': 'red',
                    'H': 'black',
                    'F': 'lightblue'
                }.get(tile, 'white')
                ax.add_patch(patches.Rectangle((j, n_rows - i - 1), 1, 1, facecolor=color, edgecolor='gray'))
                ax.text(j + 0.5, n_rows - i - 0.5, tile, ha='center', va='center', fontsize=8)

        if not failure:
            for k in range(step_idx + 1):
                idx = path[k]
                r, c = divmod(idx, n_cols)
                ax.add_patch(patches.Rectangle((c, n_rows - r - 1), 1, 1, facecolor='yellow', edgecolor='orange'))

        title = "No Path Found" if failure else f"Step {step_idx + 1}"
        ax.set_title(title)
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        frame_path = f"temp_frame_{step_idx}.png"
        plt.savefig(frame_path)
        plt.close()
        frames.append(imageio.imread(frame_path))
        os.remove(frame_path)

    imageio.mimsave(gif_name, frames, duration=0.5)
    print(f"GIF saved: {gif_name}")

def batch_ida_star_tests(num_runs=5):
    os.makedirs("gifs", exist_ok=True)
    times = []
    success_runs = 0

    for i in range(num_runs):
        print(f"\n🔁 Run {i + 1} ------------------------------")
        desc = generate_random_map(size=8)
        print("Map:")
        for row in desc:
            print("".join(row))

        env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
        start_time = time.time()
        path = ida_star(env)
        total_time = time.time() - start_time

        gif_name = f"gifs/ida_star_run_{i + 1}.gif"
        times.append(total_time if path else None)

        if path:
            success_runs += 1
            steps = len(path)
            final_pos = divmod(path[-1], 8)
            coords = [divmod(p, 8) for p in path]

            print(f"Path found by IDA* (length = {steps}):")
            print(" -> " + " -> ".join([f"({r}, {c})" for r, c in coords]))

            print(f"\nReward: 1.0")
            print(f"Time Taken: {total_time:.4f} seconds")
            print(f"Point of Convergence (steps): {steps}")

            visualize_path(path, desc, gif_name=gif_name)
        else:
            print(f"No path found in {total_time:.4f} seconds.")
            print(f"Reward: 0.0")
            print(f"Point of Convergence: -")
            visualize_path([], desc, gif_name=gif_name, failure=True)

    successful_times = [t for t in times if t is not None]
    average_time = sum(successful_times) / len(successful_times) if successful_times else 0

    plt.figure(figsize=(10, 5))
    bars = [t if t is not None else 0 for t in times]
    labels = [f"Run {i+1}" for i in range(num_runs)]
    bar_colors = ['green' if t is not None else 'red' for t in times]

    plt.bar(labels, bars, color=bar_colors)
    plt.axhline(average_time, color='blue', linestyle='--', label=f'Avg: {average_time:.4f}s')
    plt.title("Time Taken per Run (IDA*)")
    plt.xlabel("Run")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ida_star_average_time.png")
    plt.show()

    print(f"\nSuccessful runs: {success_runs}/{num_runs}")
    print(f"Average Time (successful runs only): {average_time:.4f} seconds")

if __name__ == "__main__":
    batch_ida_star_tests(num_runs=5)
from shutil import which
import matplotlib.pyplot as plt
import pickle as pk
import numpy as np

def plot_policy(optimal_policy_grid, algorithm, grid_w, grid_h):
    horizontal_min, horizontal_max, horizontal_stepsize = 0, grid_w, 1
    vertical_min, vertical_max, vertical_stepsize = 0, grid_h, -1

    xv, yv = np.meshgrid(np.arange(horizontal_min, horizontal_max, horizontal_stepsize), 
                        np.arange(vertical_max, vertical_min, vertical_stepsize))

    fig, ax = plt.subplots()

    xd, yd = np.gradient(optimal_policy_grid)

    def func_to_vectorize(x, y, optimal_policy_grid):
        # UP
        if optimal_policy_grid == 0:
            ax.arrow(x + horizontal_stepsize/2, y + vertical_stepsize/2, 0, 0.35, fc="k", ec="k", head_width=0.1, head_length=0.1)
        # Left
        if optimal_policy_grid == 1:
            ax.arrow(x + horizontal_stepsize/2, y + vertical_stepsize/2, -0.35, 0, fc="k", ec="k", head_width=0.1, head_length=0.1)
        # Down
        if optimal_policy_grid == 2:
            ax.arrow(x + horizontal_stepsize/2, y + vertical_stepsize/2, 0, -0.35, fc="k", ec="k", head_width=0.1, head_length=0.1)
        # Right
        if optimal_policy_grid == 3:
            ax.arrow(x + horizontal_stepsize/2, y + vertical_stepsize/2, 0.35, 0, fc="k", ec="k", head_width=0.1, head_length=0.1)

    vectorized_arrow_drawing = np.vectorize(func_to_vectorize)
    vectorized_arrow_drawing(xv, yv, optimal_policy_grid)
    fig.set_size_inches(grid_w, grid_h)
    plt.yticks(np.arange(0,grid_h+0.1,1))
    plt.xticks(np.arange(0,grid_w+0.1,1))
    # Place Start Point
    plt.text(0.4,0.4,'S', fontsize=20, color = 'r')
    # Place Goal Point
    plt.text(grid_w-0.6, 0.4 ,'G', fontsize=20, color = 'g')

    cliff_points = [(5,i) for i in range(1,grid_w-1)]+[(0,i) for i in range(1,grid_w-1)]
    for (row, col) in cliff_points:
        ax.add_patch(plt.Rectangle((col, row), 1, 1, color="red", alpha=0.3))

    wind_points=[(1, i) for i in range(1, 6)]
    for (row, col) in wind_points:
        ax.add_patch(plt.Rectangle((col, row), 1, 1, color="blue", alpha=0.3))

    plt.grid(which='major')
    plt.title(f'$\epsilon$-greedy Optimal Policy Learned by {algorithm}')
    #plt.show()
    plt.savefig(f'./data/readme_pics/{algorithm}_policy_map.jpg')

def plot_reward_sum(algorithm_ls, all_reward_sums):
    # Create a subplot for each algorithm
        fig, axs = plt.subplots(len(algorithm_ls), 1, figsize=(8, len(algorithm_ls) * 4), sharex=True)
    
    # Loop through each algorithm and its respective subplot
        for i, algo in enumerate(algorithm_ls):
            axs[i].plot(all_reward_sums[algo], label=algo)
            axs[i].legend()
            axs[i].set_ylim([-30, 0])
            axs[i].set_title(f'Reward Sum for {algo}')
        
    # Set a common x-axis label
        plt.xlabel("Episodes")
    # Adjust layout to avoid overlap
        plt.tight_layout()
    # Save the figure
        plt.savefig('./data/readme_pics/reward_sum_comp.jpg')


def plot_avg_reward(algorithm_ls, all_reward_sums, window=50):
    """
    Plots and saves the moving average of rewards per episode.

    Args:
        algorithm_ls: List of algorithm names.
        all_reward_sums: Dictionary of reward sums, one list per algorithm.
        window: Window size for moving average.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for algo in algorithm_ls:
        # Compute moving average of rewards for smoothing
        rewards = np.array(all_reward_sums[algo])
        avg_rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(avg_rewards, label=f'{algo} (window={window})')

    ax.set_title('Average Reward Per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.legend()
    plt.savefig(f'./data/readme_pics/avg_reward_per_episode.jpg')
    plt.close(fig)



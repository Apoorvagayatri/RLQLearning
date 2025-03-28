from agent.agent import QLearningAgent, DoubleQLearningAgent, TripleQLearningAgent,QuadrupleQLearningAgent
from environment.environment import CliffWalkEnvironment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plot import plot_policy, plot_reward_sum,plot_avg_reward

def optimal_policy(env):
    """
    Define walk along cliff optimal policy, (# of states, # of actions)
    Returns:
        policy grid
    """
    policy_ls = np.ones((env.grid_h * env.grid_w, 4)) * 0.25
    policy_ls[36] = [1,0,0,0]
    for i in range(24, 24 + env.grid_w-1):
        policy_ls[i] = [0,0,0,1]
    policy_ls[35] = [0,0,1,0]
    return policy_ls

def safe_policy(env):
    """
    Define safe policy, (# of states, # of actions)
    Returns:
        policy grid
    """
    policy_ls = np.ones((env.grid_h * env.grid_w, 4)) * 0.25
    # Go up
    for i in range(12,37,12):
        policy_ls[i] = [1, 0, 0, 0]
    # Go right
    for i in range(0, 12):
        policy_ls[i] = [0, 0, 0, 1]
    # Go down
    for i in range(11,36,12):
        policy_ls[i] = [0, 0, 1, 0]
    return policy_ls



def TD_control(agents_ls, num_episode=5000):
    all_reward_sums = {}
    for algo in algorithm_ls:
        if algo == 'QLearning':
            agent = QLearningAgent()
        if algo == 'DoubleQLearning':
            agent = DoubleQLearningAgent()
        if algo == 'TripleQLearning':
            agent=TripleQLearningAgent()
        if algo =='QuadrupleQLearning':
            agent=QuadrupleQLearningAgent()

        grid_height = 6
        grid_width = 7
        env.env_init({ "grid_height": grid_height, "grid_width": grid_width })
        agent.agent_init({"num_actions": 4, "num_states": grid_height * grid_width, "epsilon": 0.1, "step_size": 0.5, "discount": 1.0})

        all_reward_sums[algo] = []
        for _ in range(num_episode):
            reward_sum = 0
            all_iterations=0
            # Start episode
            state = env.env_start()
            action = agent.agent_start(state)
            reward, state, terminal = env.env_step(action)
            reward_sum += reward
            while not terminal:
                all_iterations+=1
                action = agent.agent_step(reward, state)
                reward, state, terminal = env.env_step(action)
                reward_sum += reward
            else:
                agent.agent_end(reward)
                env.env_cleanup()
                all_reward_sums[algo].append(reward_sum)

            print(f"Episode: {_}, State: {state}, Action: {action}, Reward: {reward}")

        # Determine optimal policy map based on the number of Q-tables in each agent
        if hasattr(agent, 'q1') and hasattr(agent, 'q2') and hasattr(agent, 'q3') and hasattr(agent, 'q4'):
    # Quadruple Q-Learning: Average over q1, q2, q3, and q4
            optimal_policy_map = np.argmax((agent.q1 + agent.q2 + agent.q3 + agent.q4) / 4, axis=1).reshape((env.grid_h, env.grid_w))
        elif hasattr(agent, 'q1') and hasattr(agent, 'q2') and hasattr(agent, 'q3'):
    # Triple Q-Learning: Average over q1, q2, and q3
            optimal_policy_map = np.argmax((agent.q1 + agent.q2 + agent.q3) / 3, axis=1).reshape((env.grid_h, env.grid_w))
        elif hasattr(agent, 'q1') and hasattr(agent, 'q2'):
    # Double Q-Learning: Average over q1 and q2
            optimal_policy_map = np.argmax((agent.q1 + agent.q2) / 2, axis=1).reshape((env.grid_h, env.grid_w))
        else:
    # Single Q-Learning agent
            optimal_policy_map = np.argmax(agent.q, axis=1).reshape((env.grid_h, env.grid_w))

        plot_policy(optimal_policy_map, algo, env.grid_w, env.grid_h)
        
    plot_reward_sum(algorithm_ls, all_reward_sums)
    plot_avg_reward(algorithm_ls, all_reward_sums)

    
if __name__ == "__main__":
    env = CliffWalkEnvironment()
    num_episode = 5000
    
    # TD prediction
    #TD_zero(optimal_policy, num_episode)
    #TD_zero(safe_policy, num_episode)
    
    # TD control
    algorithm_ls = ['QLearning', 'DoubleQLearning','TripleQLearning','QuadrupleQLearning']
    TD_control(algorithm_ls, num_episode)
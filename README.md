# Reinforcement Learning - Advanced Q-Learning Algorithms

A comprehensive implementation of various Q-Learning algorithms for the Cliff Walking environment.

> **Note:** This project builds upon the base implementation from [John-CYHui/Reinforcement-Learning-Cliff-Walking](https://github.com/John-CYHui/Reinforcement-Learning-Cliff-Walking).

## Project Overview

This project implements and compares multiple Q-Learning variants in a Cliff Walking environment, including:
- Standard Q-Learning
- Double Q-Learning
- Triple Q-Learning
- Quadruple Q-Learning

The goal is to analyze how these different algorithms perform in terms of convergence, stability, and overall performance in a dynamic reinforcement learning environment.

## Environment

The Cliff Walking environment is a grid-world where an agent must navigate from a starting position to a goal while avoiding falling off a cliff. The environment includes:

- Grid dimensions customizable (default: 6×7)
- Starting position at the bottom-left corner
- Goal position at the bottom-right corner
- Cliff along specific positions that result in a large negative reward (-100)
- Regular step penalty (-1) to encourage efficient paths

## Algorithms

### Q-Learning
Standard Q-Learning algorithm using a single Q-table to estimate action values.

### Double Q-Learning
Uses two Q-tables to reduce overestimation bias in value functions.

### Triple Q-Learning
An extension using three Q-tables for even more stable value estimation.

### Quadruple Q-Learning
Further extends the concept to four Q-tables for maximum stability.

## Project Structure

```
.
├── agent/
│   └── agent.py                # Implementation of all agent algorithms
├── environment/
│   └── environment.py          # Cliff Walking environment implementation
├── data/                       # Data storage directory
├── abstract_classes.py         # Base classes for agents and environments
├── main.py                     # Main execution script
├── plot.py                     # Visualization utilities
└── docs/
    └── observations_report.pdf # Detailed analysis of algorithm performance
```

## Key Features

- Modular design with separated environment and agent components
- Dynamic policy visualization
- Comprehensive performance metrics
- Comparative analysis across algorithms

## Running the Project

To run the project with default settings:

```bash
python main.py
```

This will:
1. Initialize the Cliff Walking environment
2. Train all four Q-Learning variants for 5000 episodes
3. Generate policy visualizations
4. Plot reward summaries and average rewards for comparison

## Observations Report

For an in-depth analysis of the algorithms' performance, refer to the `docs/observations_report.pdf` which contains:

- Detailed comparison of all four Q-Learning variants
- Stability analysis across different hyperparameters
- Convergence metrics and visualization
- Trade-offs between the algorithms
- Recommendations for practical applications

## Parameters

The default parameters used in the experiments:
- Number of episodes: 5000
- Learning rate (α): 0.5
- Discount factor (γ): 1.0
- Exploration rate (ε): 0.1

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## License

This project is provided for educational purposes. 

## Acknowledgments

- Base implementation adapted from [John-CYHui/Reinforcement-Learning-Cliff-Walking](https://github.com/John-CYHui/Reinforcement-Learning-Cliff-Walking)
- Based on the cliff walking problem from Reinforcement Learning: An Introduction by Sutton & Barto (Example 6.6) 
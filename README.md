# Fire-Rescue Reinforcement Learning Agent

A comprehensive reinforcement learning project implementing and comparing multiple RL algorithms (DQN, REINFORCE, PPO, A2C) on a custom Fire-Rescue environment.

[Watch Demo & Explanation Video]()
[Link to Trained Models]()

## Project Overview

This project implements an autonomous agent that navigates a building environment to locate and rescue survivors. The agent must:

- Navigate through a 10x10 grid with static obstacles (walls)
- Use scanning to locate survivors
- Pick up survivors and transport them to the exit
- Minimize time and avoid collisions

## Problem Statement

**Scenario:** A rescue agent must operate in a building environment with obstacles to find and rescue a survivor.

**Objective:** Train an intelligent agent that can efficiently:

1. Explore the environment
2. Locate the survivor
3. Navigate to the survivor's location
4. Pick up the survivor
5. Return to the exit door
6. Drop-off the survivor
7. Successfully complete the rescue

**Challenges:**

- Limited visibility (scanning required)
- Static obstacles (walls)
- Time constraints (200 steps maximum)
- Action space coordination (movement + special actions)

## Environment Specifications

### State Space (8 dimensions)

- Agent X position (normalized)
- Agent Y position (normalized)
- Survivor X position (normalized)
- Survivor Y position (normalized)
- Carrying flag (0 or 1)
- Distance to survivor (normalized)
- Distance to door (normalized)
- Time remaining (normalized)

### Action Space (6 discrete actions)

- **0:** Move UP (North)
- **1:** Move DOWN (South)
- **2:** Move LEFT (West)
- **3:** Move RIGHT (East)
- **4:** SCAN (detect nearby survivor)
- **5:** PICKUP/DROP (interact with survivor)

### Reward Structure

- **-0.01** per step (encourages efficiency)
- **-1.0** for wall collision or out-of-bounds
- **+0.5** for successful scan near survivor
- **-0.05** for unsuccessful scan
- **+10.0** for picking up survivor
- **+20.0** for dropping off survivor at door
- **+30.0** bonus for mission success
- **-10.0** for timeout

### Termination Conditions

- **Success:** Survivor delivered to exit door and droped-off
- **Failure:** Time limit exceeded (200 steps)

## Implemented Algorithms

### 1. DQN (Deep Q-Network) - Value-Based

- Experience replay buffer
- Target network for stability
- Epsilon-greedy exploration
- **Best Config:** config_4_large_batch

### 2. REINFORCE - Policy Gradient

- Monte Carlo policy gradient
- Entropy regularization
- Return normalization
- **Best Config:** best_model

### 3. PPO (Proximal Policy Optimization) - Policy Gradient

- Clipped objective function
- Generalized Advantage Estimation (GAE)
- Multiple epochs per batch
- **Best Config:** config_7_high_gamma

### 4. A2C (Advantage Actor-Critic) - Actor-Critic

- Synchronous advantage estimation
- Value function baseline
- Entropy bonus for exploration
- **Best Config:** config_4_low_gamma

## 📊 Hyperparameter Tuning

Each algorithm was trained with **10+ different configurations** to find optimal hyperparameters:

### Key Parameters Tested:

- Learning rates: 1e-5 to 5e-3
- Discount factors (gamma): 0.95 to 0.995
- Batch sizes: 32 to 256
- Network architectures: [64,64] to [128,128]
- Exploration parameters (algorithm-specific)

### Training Details:

- **100,000 timesteps** per configuration
- **Total training time:** ~20-40 hours
- **Evaluation:** 50-100 episodes per configuration
- **Metrics tracked:** Reward, success rate, efficiency, collision rate

## Installation & Usage

### Prerequisites

```bash
Python 3.8+
pip
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/fire_rescue_agent.git
cd fire_rescue_agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Main Comparison and visualization Script

```bash
# Compare all trained models and visualize the best trained model
python main.py
```

This will:

1. Load all best trained models (PPO, A2C, DQN, REINFORCE)
2. Evaluate each on 50 episodes
3. Compare performance metrics
4. Offer to visualize the best model in action

### Training Individual Models

```bash
# Training notebooks are in the training/ directory
# Open in Jupyter or VS Code:
jupyter notebook training/ppo_training.ipynb
```

### Running Random Agent Demo

This will visualize the custom environment + a random untrained agent:

```bash
python environment/random_agent_demo.py
```

## Project Structure

```
fire_rescue_agent/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py          # Gymnasium environment
│   ├── rendering.py            # Pygame visualization
│   ├── random_agent_demo.py    # Random baseline demo
│   └── test_environment.ipynb  # Environment testing
├── training/
│   ├── dqn_training.ipynb      # DQN training & tuning
│   ├── reinforcement_training.ipynb  # REINFORCE training
│   ├── ppo_training.ipynb      # PPO training & tuning
│   └── a2c_training.ipynb      # A2C training & tuning
├── models/
│   ├── dqn/                    # Trained DQN models
│   ├── reinforce/              # Trained REINFORCE models
│   ├── ppo/                    # Trained PPO models
│   └── a2c/                    # Trained A2C models
├── results/
│   └── plots/                  # Training plots & analysis
├── main.py                     # Model comparison script and visualization
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── ASSIGNMENT_CHECKLIST.md     # Completion checklist
```

## 📈 Results Summary

| Algorithm | Mean Reward    | Success Rate | Avg Steps | Best Config          |
| --------- | -------------- | ------------ | --------- | -------------------- |
| DQN       | 56.91 ± 51.05  | 62%          | 200.0     | config_4_large_batch |
| A2C       | 50.50 ± 78.31  | 35%          | 200.0     | config_4_low_gamma   |
| PPO       | 18.66 ± 125.97 | 73%          | 200.0     | config_7_high_gamma  |
| REINFORCE | Varies         | Varies       | ~180      | best_model           |

_(Run `python main.py` for complete evaluation)_

## Visualization Features

The Pygame visualization includes:

- **Real-time rendering** of agent, survivor, walls, and exit
- **Animated effects** (agent pulse, danger indicators)
- **Information panel** with live metrics
- **Color-coded states** (carrying vs. searching)
- **Visual feedback** for all actions
- **Performance metrics** display

## Key Findings

### Algorithm Comparison:

1. **PPO** showed best sample efficiency with stable learning
2. **DQN** achieved highest average rewards but more variance
3. **A2C** balanced performance with fast training
4. **REINFORCE** required more tuning but provided policy insights

### Important Insights:

- **Exploration crucial:** Entropy regularization significantly improved performance
- **Reward shaping removed:** Distance-based shaping caused exploitation; removed for better learning
- **Gamma matters:** Higher discount factors (0.99-0.995) performed better for long-horizon task
- **Batch size impact:** Larger batches (128-256) provided more stable gradients

## Technical Details

### Libraries Used:

- **Gymnasium** - Environment interface
- **Stable-Baselines3** - DQN, PPO, A2C implementations
- **PyTorch** - REINFORCE implementation
- **Pygame** - Visualization
- **NumPy** - Numerical operations
- **Matplotlib/Seaborn** - Result visualization

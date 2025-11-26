"""
Generalization Test for Fire-Rescue RL Agents

Tests trained models on unseen initial states with different random seeds
to evaluate generalization capability.

Author: Fire-Rescue RL Project
Date: November 2025
"""

import sys
import os
import numpy as np
import json
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add environment to path
sys.path.insert(0, os.path.dirname(__file__))

from environment.custom_env import FireRescueEnv
from stable_baselines3 import PPO, A2C, DQN

# Set plotting style
sns.set_style("whitegrid")

# ═══════════════════════════════════════════════════════════════════════════
# REINFORCE POLICY NETWORK (for loading saved model)
# ═══════════════════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE agent."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.0))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


class REINFORCEAgent:
    """REINFORCE agent for inference."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
    
    def select_action(self, state, deterministic=True):
        """Select action using policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy(state)
            if deterministic:
                return torch.argmax(probs).item()
            else:
                from torch.distributions import Categorical
                m = Categorical(probs)
                return m.sample().item()
    
    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()


# ═══════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

MODELS = {
    "DQN": {
        "path": "models/dqn/config_10_optimized/final_model.zip",
        "algorithm": "DQN"
    },
    "PPO": {
        "path": "models/ppo/config_7_high_gamma/final_model.zip",
        "algorithm": "PPO"
    },
    "A2C": {
        "path": "models/a2c/config_4_low_gamma/config_4_low_gamma_model.zip",
        "algorithm": "A2C"
    },
    "REINFORCE": {
        "path": "models/reinforce/best_model/best_reinforce.pth",
        "algorithm": "REINFORCE"
    }
}

# Generalization test parameters
NUM_TEST_EPISODES = 100  # Test on 100 unseen scenarios
SEED_OFFSET = 10000  # Use seeds 10000-10099 (different from training)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_info):
    """Load a trained model based on algorithm type."""
    algorithm = model_info["algorithm"]
    path = model_info["path"]
    
    if not os.path.exists(path):
        print(f"❌ ERROR: Model not found at {path}")
        return None
    
    try:
        if algorithm == "PPO":
            model = PPO.load(path)
        elif algorithm == "A2C":
            model = A2C.load(path)
        elif algorithm == "DQN":
            model = DQN.load(path)
        elif algorithm == "REINFORCE":
            env = FireRescueEnv(grid_size=10, max_time=250)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            env.close()
            
            model = REINFORCEAgent(state_dim, action_dim, hidden_dims=[128, 64])
            model.load(path)
        else:
            print(f"❌ ERROR: Unknown algorithm '{algorithm}'")
            return None
        
        print(f"✅ Loaded {algorithm} model")
        return model
    
    except Exception as e:
        print(f"❌ ERROR loading {algorithm} model: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# GENERALIZATION TEST FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def test_generalization(model, model_name, algorithm, num_episodes=100, seed_offset=10000):
    """
    Test model generalization on unseen initial states.
    
    Args:
        model: Trained model
        model_name: Name of the model
        algorithm: Algorithm type
        num_episodes: Number of test episodes
        seed_offset: Seed offset to ensure different initial states
        
    Returns:
        Dictionary with test metrics
    """
    print(f"\n{'='*80}")
    print(f"Testing Generalization: {model_name} ({algorithm})")
    print(f"{'='*80}")
    print(f"Testing on {num_episodes} unseen scenarios (seeds {seed_offset}-{seed_offset+num_episodes-1})")
    
    env = FireRescueEnv(grid_size=10, max_time=250)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    wall_collision_rates = []
    scan_efficiencies = []
    pickup_attempts_list = []
    time_to_find_survivor_list = []
    
    for episode in range(num_episodes):
        # Use different seed for each episode
        test_seed = seed_offset + episode
        obs, _ = env.reset(seed=test_seed)
        
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Get action from model
            if algorithm == "REINFORCE":
                action = model.select_action(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if info.get('success', False):
            success_count += 1
        
        wall_collision_rates.append(info.get('wall_collision_rate', 0))
        scan_efficiencies.append(info.get('scan_efficiency', 0))
        pickup_attempts_list.append(info.get('pickup_attempts', 0))
        
        time_found = info.get('time_to_find_survivor', None)
        if time_found is not None:
            time_to_find_survivor_list.append(time_found)
        
        # Print progress every 20 episodes
        if (episode + 1) % 20 == 0:
            print(f"   Episodes {episode + 1}/{num_episodes} completed...")
    
    env.close()
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = success_count / num_episodes
    mean_wall_collisions = np.mean(wall_collision_rates)
    mean_scan_efficiency = np.mean(scan_efficiencies)
    mean_pickup_attempts = np.mean(pickup_attempts_list)
    mean_time_to_find = np.mean(time_to_find_survivor_list) if time_to_find_survivor_list else None
    
    # Print results
    print(f"\n📊 Generalization Test Results for {model_name}:")
    print(f"   Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   Success Rate: {success_rate:.1%} ({success_count}/{num_episodes})")
    print(f"   Avg Episode Length: {mean_length:.1f} steps")
    print(f"   Avg Wall Collision Rate: {mean_wall_collisions:.2%}")
    print(f"   Avg Scan Efficiency: {mean_scan_efficiency:.2%}")
    print(f"   Avg Pickup Attempts: {mean_pickup_attempts:.2f}")
    if mean_time_to_find is not None:
        print(f"   Avg Time to Find Survivor: {mean_time_to_find:.1f} steps")
    
    return {
        "model_name": model_name,
        "algorithm": algorithm,
        "num_test_episodes": num_episodes,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "success_count": success_count,
        "mean_episode_length": mean_length,
        "mean_wall_collision_rate": mean_wall_collisions,
        "mean_scan_efficiency": mean_scan_efficiency,
        "mean_pickup_attempts": mean_pickup_attempts,
        "mean_time_to_find_survivor": mean_time_to_find,
        "episode_rewards": episode_rewards,
    }


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def visualize_generalization_results(results_list, output_path="results/plots/generalization_test.png"):
    """Create visualization comparing generalization performance."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Generalization Test Results (100 Unseen Scenarios)', fontsize=16, fontweight='bold')
    
    methods = [r['algorithm'] for r in results_list]
    mean_rewards = [r['mean_reward'] for r in results_list]
    std_rewards = [r['std_reward'] for r in results_list]
    success_rates = [r['success_rate'] * 100 for r in results_list]
    collision_rates = [r['mean_wall_collision_rate'] * 100 for r in results_list]
    
    colors = {'DQN': 'steelblue', 'PPO': 'orange', 'A2C': 'green', 'REINFORCE': 'purple'}
    bar_colors = [colors.get(m, 'gray') for m in methods]
    
    # Mean Rewards
    ax = axes[0, 0]
    bars = ax.bar(methods, mean_rewards, yerr=std_rewards, color=bar_colors, alpha=0.7, 
                   capsize=5, edgecolor='black', linewidth=1.5)
    ax.set_title('Mean Reward on Unseen States', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    for bar, value in zip(bars, mean_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}',
                ha='center', va='bottom' if value > 0 else 'top', fontsize=10, fontweight='bold')
    
    # Success Rates
    ax = axes[0, 1]
    bars = ax.bar(methods, success_rates, color=bar_colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    ax.set_title('Success Rate on Unseen States', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Reward Distribution (box plot)
    ax = axes[1, 0]
    reward_data = [r['episode_rewards'] for r in results_list]
    bp = ax.boxplot(reward_data, labels=methods, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], bar_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title('Reward Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Wall Collision Rates
    ax = axes[1, 1]
    bars = ax.bar(methods, collision_rates, color=bar_colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    ax.set_title('Wall Collision Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('Collision Rate (%)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, value in zip(bars, collision_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualization saved: {output_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("🚒 FIRE-RESCUE AGENT - GENERALIZATION TEST")
    print("="*80)
    print(f"\nTesting models on {NUM_TEST_EPISODES} unseen scenarios")
    print(f"Using random seeds: {SEED_OFFSET} to {SEED_OFFSET + NUM_TEST_EPISODES - 1}")
    print("="*80 + "\n")
    
    # Load all models
    loaded_models = {}
    for name, info in MODELS.items():
        model = load_model(info)
        if model is not None:
            loaded_models[name] = {
                "model": model,
                "info": info
            }
    
    if len(loaded_models) == 0:
        print("\n❌ ERROR: No models could be loaded. Exiting.")
        return
    
    print(f"\n✅ Successfully loaded {len(loaded_models)}/{len(MODELS)} models\n")
    
    # Test generalization for all models
    generalization_results = []
    for name, model_data in loaded_models.items():
        result = test_generalization(
            model_data["model"],
            name,
            model_data["info"]["algorithm"],
            NUM_TEST_EPISODES,
            SEED_OFFSET
        )
        generalization_results.append(result)
    
    # Save results
    output_file = "generalization_results.json"
    # Remove episode_rewards from saved results (too large)
    save_results = []
    for r in generalization_results:
        r_copy = r.copy()
        r_copy.pop('episode_rewards', None)
        save_results.append(r_copy)
    
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=4)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Create visualization
    visualize_generalization_results(generalization_results)
    
    # Print summary comparison
    print("\n" + "="*80)
    print("📊 GENERALIZATION SUMMARY")
    print("="*80)
    print(f"\n{'Algorithm':<15} {'Success Rate':<15} {'Mean Reward':<20} {'Std Reward':<15}")
    print("-" * 80)
    for result in generalization_results:
        print(f"{result['algorithm']:<15} {result['success_rate']:>6.1%}          "
              f"{result['mean_reward']:>8.2f}           {result['std_reward']:>8.2f}")
    print("="*80 + "\n")
    
    print("✅ GENERALIZATION TEST COMPLETED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

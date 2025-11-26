"""
Fire-Rescue Agent - Main Comparison and Visualization Script

This script:
1. Loads the best models from each RL algorithm (PPO, A2C, DQN, REINFORCE)
2. Evaluates all models on the Fire-Rescue environment
3. Compares their performance metrics
4. Visualizes the best-performing agent in action

Author: Fire-Rescue RL Project
Date: November 2025
"""

import sys
import os
import numpy as np
import json
import time
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Add environment to path
sys.path.insert(0, os.path.dirname(__file__))

from environment.custom_env import FireRescueEnv
from environment.rendering import FireRescueRenderer
from stable_baselines3 import PPO, A2C, DQN

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

MODELS = {
    "PPO": {
        "path": "models/ppo/config_7_high_gamma/final_model.zip",
        "config": "config_7_high_gamma",
        "algorithm": "PPO"
    },
    "A2C": {
        "path": "models/a2c/config_4_low_gamma/config_4_low_gamma_model.zip",
        "config": "config_4_low_gamma",
        "algorithm": "A2C"
    },
    "DQN": {
        "path": "models/dqn/config_4_large_batch/final_model.zip",
        "config": "config_4_large_batch",
        "algorithm": "DQN"
    },
    "REINFORCE": {
        "path": "models/reinforce/best_model/best_reinforce.pth",
        "config": "best_model",
        "algorithm": "REINFORCE"
    }
}

NUM_EVAL_EPISODES = 50  # Number of episodes to evaluate each model
VISUALIZATION_EPISODES = 5  # Number of episodes to visualize for best model

# ═══════════════════════════════════════════════════════════════════════════
# REINFORCE POLICY NETWORK (Same architecture as training)
# ═══════════════════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE agent."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        super(PolicyNetwork, self).__init__()
        
        # Build network matching the saved checkpoint structure
        # Saved model has layer indices: 0, 3, 6 (Linear layers) with ReLU and Dropout in between
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.0))  # Dropout with p=0 (maintains layer indices)
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
                m = Categorical(probs)
                return m.sample().item()
    
    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_info):
    """
    Load a trained model based on algorithm type.
    
    Args:
        model_info: Dictionary containing model path and algorithm info
        
    Returns:
        Loaded model object
    """
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
            # Load custom REINFORCE model
            env = FireRescueEnv()
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            env.close()
            
            model = REINFORCEAgent(state_dim, action_dim, hidden_dims=[128, 64])
            model.load(path)
        else:
            print(f"❌ ERROR: Unknown algorithm '{algorithm}'")
            return None
        
        print(f"✅ Loaded {algorithm} model from {path}")
        return model
    
    except Exception as e:
        print(f"❌ ERROR loading {algorithm} model: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, model_name, algorithm, num_episodes=50):
    """
    Evaluate a model on the Fire-Rescue environment.
    
    Args:
        model: Trained model
        model_name: Name of the model
        algorithm: Algorithm type (for REINFORCE special handling)
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name} ({algorithm})")
    print(f"{'='*80}")
    
    env = FireRescueEnv()
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    wall_collision_rates = []
    scan_efficiencies = []
    pickup_attempts_list = []
    time_to_find_survivor_list = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
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
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
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
    print(f"\n📊 Results for {model_name}:")
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
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
        "success_count": success_count,
        "mean_episode_length": mean_length,
        "mean_wall_collision_rate": mean_wall_collisions,
        "mean_scan_efficiency": mean_scan_efficiency,
        "mean_pickup_attempts": mean_pickup_attempts,
        "mean_time_to_find_survivor": mean_time_to_find,
        "num_episodes": num_episodes
    }


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def visualize_agent(model, model_name, algorithm, num_episodes=3):
    """
    Visualize the agent in action using pygame rendering.
    
    Args:
        model: Trained model
        model_name: Name of the model
        algorithm: Algorithm type
        num_episodes: Number of episodes to visualize
    """
    print(f"\n{'='*80}")
    print(f"🎮 VISUALIZING: {model_name} ({algorithm})")
    print(f"{'='*80}")
    print(f"\n🎬 Running {num_episodes} episodes with GUI visualization...")
    print(f"   Press ESC or close window to stop early\n")
    
    env = FireRescueEnv()
    renderer = FireRescueRenderer(grid_size=env.grid_size, cell_size=60)
    
    for episode in range(num_episodes):
        print(f"\n{'─'*80}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'─'*80}")
        
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            # Render current state
            renderer.render(env, step_count, episode_reward, episode + 1, info)
            
            # Handle pygame events (check for quit)
            if not renderer.handle_events():
                print("\n❌ Visualization stopped by user")
                renderer.close()
                env.close()
                return
            
            # Get action from model
            if algorithm == "REINFORCE":
                action = model.select_action(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Small delay for visibility
            time.sleep(0.1)
        
        # Final render
        renderer.render(env, step_count, episode_reward, episode + 1, info)
        time.sleep(1.0)  # Pause at end of episode
        
        # Print episode summary
        print(f"\n✅ Episode {episode + 1} Complete:")
        print(f"   Total Reward: {episode_reward:.2f}")
        print(f"   Steps: {step_count}")
        print(f"   Success: {'YES ✓' if info.get('success', False) else 'NO ✗'}")
        print(f"   Wall Collisions: {info.get('wall_collision_rate', 0):.1%}")
        print(f"   Scan Efficiency: {info.get('scan_efficiency', 0):.1%}")
        if info.get('time_to_find_survivor') is not None:
            print(f"   Time to Find Survivor: {info.get('time_to_find_survivor')} steps")
    
    print(f"\n{'='*80}")
    print(f"Visualization Complete!")
    print(f"{'='*80}\n")
    
    renderer.close()
    env.close()


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON & REPORT
# ═══════════════════════════════════════════════════════════════════════════

def compare_and_rank_models(results):
    """
    Compare all models and determine the best performer.
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        Sorted list of results (best to worst)
    """
    print(f"\n{'='*80}")
    print(f"📊 MODEL COMPARISON REPORT")
    print(f"{'='*80}\n")
    
    # Sort by mean reward (primary metric)
    sorted_results = sorted(results, key=lambda x: x['mean_reward'], reverse=True)
    
    # Print comparison table
    print(f"{'Rank':<6} {'Algorithm':<12} {'Mean Reward':<18} {'Success Rate':<15} {'Avg Steps':<12} {'Scan Eff':<12}")
    print(f"{'='*80}")
    
    for i, result in enumerate(sorted_results, 1):
        rank_icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{rank_icon} #{i:<4} {result['algorithm']:<12} "
              f"{result['mean_reward']:>7.2f} ± {result['std_reward']:<5.2f}  "
              f"{result['success_rate']:>6.1%}          "
              f"{result['mean_episode_length']:>7.1f}     "
              f"{result['mean_scan_efficiency']:>6.1%}")
    
    print(f"{'='*80}\n")
    
    # Print best model details
    best = sorted_results[0]
    print(f"🏆 BEST PERFORMING MODEL: {best['model_name']} ({best['algorithm']})")
    print(f"{'─'*80}")
    print(f"   Mean Reward:          {best['mean_reward']:.2f} ± {best['std_reward']:.2f}")
    print(f"   Success Rate:         {best['success_rate']:.1%} ({best['success_count']}/{best['num_episodes']})")
    print(f"   Avg Episode Length:   {best['mean_episode_length']:.1f} steps")
    print(f"   Wall Collision Rate:  {best['mean_wall_collision_rate']:.2%}")
    print(f"   Scan Efficiency:      {best['mean_scan_efficiency']:.2%}")
    print(f"   Avg Pickup Attempts:  {best['mean_pickup_attempts']:.2f}")
    if best['mean_time_to_find_survivor'] is not None:
        print(f"   Time to Find Survivor: {best['mean_time_to_find_survivor']:.1f} steps")
    print(f"{'─'*80}\n")
    
    return sorted_results


def save_comparison_results(results, output_path="comparison_results.json"):
    """Save comparison results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"💾 Results saved to: {output_path}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("🚒 FIRE-RESCUE AGENT - MODEL COMPARISON & VISUALIZATION")
    print("="*80)
    print(f"\nThis script will:")
    print(f"   1. Load best models from each algorithm (PPO, A2C, DQN, REINFORCE)")
    print(f"   2. Evaluate all models on {NUM_EVAL_EPISODES} episodes")
    print(f"   3. Compare and rank performance")
    print(f"   4. Visualize the best model in action")
    print("="*80 + "\n")
    
    # Step 1: Load all models
    print("📂 STEP 1: Loading Models")
    print("─"*80)
    
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
    
    # Step 2: Evaluate all models
    print("📊 STEP 2: Evaluating All Models")
    print("─"*80)
    
    evaluation_results = []
    for name, model_data in loaded_models.items():
        result = evaluate_model(
            model_data["model"],
            name,
            model_data["info"]["algorithm"],
            NUM_EVAL_EPISODES
        )
        evaluation_results.append(result)
    
    # Step 3: Compare and rank models
    print("\n📈 STEP 3: Comparing Models")
    print("─"*80)
    
    sorted_results = compare_and_rank_models(evaluation_results)
    
    # Save results
    save_comparison_results(sorted_results, "comparison_results.json")
    
    # Step 4: Visualize best model
    print("🎬 STEP 4: Visualizing Best Model")
    print("─"*80)
    
    best_model_name = sorted_results[0]["model_name"]
    best_model_data = loaded_models[best_model_name]
    
    user_input = input(f"\nVisualize {best_model_name} in action? (y/n): ").strip().lower()
    
    if user_input == 'y' or user_input == 'yes':
        visualize_agent(
            best_model_data["model"],
            best_model_name,
            best_model_data["info"]["algorithm"],
            VISUALIZATION_EPISODES
        )
    else:
        print("\nSkipping visualization.\n")
    
    # Final summary
    print("="*80)
    print("✅ ALL TASKS COMPLETED!")
    print("="*80)
    print(f"\n🏆 Winner: {sorted_results[0]['model_name']} ({sorted_results[0]['algorithm']})")
    print(f"   Reward: {sorted_results[0]['mean_reward']:.2f} ± {sorted_results[0]['std_reward']:.2f}")
    print(f"   Success Rate: {sorted_results[0]['success_rate']:.1%}\n")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

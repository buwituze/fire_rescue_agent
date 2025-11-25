"""
Fire-Rescue Environment - Random Agent Demonstration
Shows visualization with an UNTRAINED agent taking RANDOM actions
"""

import sys
from pathlib import Path

# Add environment to path
sys.path.append(str(Path(__file__).parent / 'environment'))

from custom_env import FireRescueEnv
from rendering import FireRescueRenderer
import numpy as np
import time

def run_demo(episodes=3, max_steps=200, render_delay=0.1):
    """
    Run random agent demonstration with visualization
    
    Args:
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render_delay: Delay between frames (seconds)
    """
    print("="*70)
    print("Fire-Rescue Environment - Random Agent Demonstration")
    print("="*70)
    print("\nVisualization: Untrained agent taking random actions")
    print("Controls: Press ESC or close window to exit\n")
    print("="*70)
    
    # Create environment and renderer
    env = FireRescueEnv(grid_size=10, max_time=200)
    renderer = FireRescueRenderer(grid_size=10, cell_size=60)
    
    for episode in range(1, episodes + 1):
        print(f"\n{'='*70}")
        print(f"Episode {episode}/{episodes}")
        print(f"{'='*70}")
        
        # Reset
        obs, info = env.reset()
        total_reward = 0
        step = 0
        done = False
        truncated = False
        
        print(f"Agent: {env.agent} | Survivor: {env.survivor} | Door: {env.door}")
        
        while not (done or truncated) and step < max_steps:
            # Random action (NO MODEL)
            action = env.action_space.sample()
            
            # Step
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Render
            renderer.render(
                env=env,
                step_count=step,
                total_reward=total_reward,
                episode=episode,
                info=info
            )
            
            # Handle quit
            if not renderer.handle_events():
                print("\nExiting...")
                renderer.close()
                return
            
            time.sleep(render_delay)
        
        # Episode summary
        result = "SUCCESS ✓" if info.get('success', False) else "FAILED ✗"
        print(f"\n{result} | Steps: {step} | Reward: {total_reward:.2f}")
        
        if episode < episodes:
            time.sleep(2)
    
    print(f"\n{'='*70}")
    print("Demo Complete - Random agent shows baseline performance")
    print("="*70)
    
    time.sleep(3)
    renderer.close()

if __name__ == "__main__":
    try:
        run_demo(episodes=3, max_steps=200, render_delay=0.1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
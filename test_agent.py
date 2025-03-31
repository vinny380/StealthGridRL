#!/usr/bin/env python3
"""
Test script for evaluating trained stealth exploration agent.
"""

import os
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from coverage_gridworld.custom import SafetyRewardWrapper

# Add coverage-gridworld to path
sys.path.append("coverage-gridworld")

# Import environment registration
import coverage_gridworld

def test_agent(model_path, map_name, num_episodes=3):
    print(f"\nTesting on {map_name} map...\n")
    
    # Create environment
    env = gym.make("safe", render_mode="human")  # Add render_mode to visualize
    env = SafetyRewardWrapper(env)
    
    # Load the trained model
    model = PPO.load(model_path)
    
    print("Observation Space Info:")
    print(f"Environment observation space: {env.observation_space}")
    print(f"Model observation space: {model.observation_space}")
    
    total_rewards = []
    total_coverages = []
    successes = 0
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        observation, info = env.reset()
        
        print(f"Initial observation shape: {observation.shape}")
        print(f"Initial observation min: {observation.min()}, max: {observation.max()}")
        
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            action, _ = model.predict(observation)
            observation, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            # Print progress every 10 steps
            if steps % 10 == 0:
                print(f"Step {steps}, Reward: {episode_reward:.2f}, Coverage: {info['total_covered_cells']}/{info['coverable_cells']}")
            
            # Add a small delay to make visualization easier to follow
            time.sleep(0.05)
        
        if 'coverage_percentage' in info:
            total_coverages.append(info['coverage_percentage'])
        if 'success' in info and info['success']:
            successes += 1
        total_rewards.append(episode_reward)
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"Total steps: {steps}")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Coverage: {info['total_covered_cells']}/{info['coverable_cells']}")
        print(f"Game over by enemy: {info['game_over']}")
        
        # Add a pause between episodes
        time.sleep(1)
    
    print("\nTest Summary:")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    if total_coverages:
        print(f"Average coverage: {np.mean(total_coverages):.1f}%")
    print(f"Success rate: {(successes / num_episodes) * 100:.1f}%")

def test_on_all_maps(model_path, num_episodes=2):
    """
    Test a trained agent on all available maps
    """
    maps = ["just_go", "safe", "maze", "chokepoint", "sneaky_enemies"]
    
    results = {}
    
    for map_name in maps:
        coverage, detection_rate = test_agent(model_path, map_name, num_episodes)
        results[map_name] = (coverage, detection_rate)
    
    # Print comparative results
    print("\nComparative Results:")
    print("-" * 50)
    print(f"{'Map':<15} {'Coverage':<10} {'Detection Rate':<15}")
    print("-" * 50)
    
    for map_name, (coverage, detection_rate) in results.items():
        print(f"{map_name:<15} {coverage:.2f}      {detection_rate:.2f}")
    
    return results

if __name__ == "__main__":
    # Test on different maps
    model_path = "models/stealth_agent_safe.zip"  # Using the stealth agent safe model
    maps = ["safe"]  # Just test one map for debugging
    
    for map_name in maps:
        print(f"\nTesting on {map_name} map...")
        test_agent(model_path, map_name)
    
    # Uncomment to test on all maps
    # test_on_all_maps(model_path) 
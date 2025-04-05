import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import from the coverage gridworld environment
sys.path.append("coverage-gridworld")
from coverage_gridworld import custom
from main import train_specific_model, run_specific_model, GridworldFeaturesExtractor, DangerAwareEnv, ObservationWrapper

# Maze with walls but no enemies
# This is based on the existing "maze" map but with enemies (value 4) removed
MAZE_NO_ENEMIES = [
    [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],  # Agent at (0,0)
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
    [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
    [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],  # Replaced enemies (4) with empty cells (0)
    [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
]

def create_custom_env(obs_space, reward_func, render_mode=None):
    """Create environment with our custom maze map"""
    # Set the observation space and reward function
    custom.set_observation_space(obs_space)
    custom.set_active_reward_function(reward_func)
    
    # Create the environment with our custom map
    env = gym.make("standard", 
                  render_mode=render_mode, 
                  predefined_map=MAZE_NO_ENEMIES, 
                  activate_game_status=False)
    
    # Apply wrappers
    env = DangerAwareEnv(env)
    env = ObservationWrapper(env, obs_space_type=obs_space)
    
    # Add monitoring
    log_dir = f"./logs/no_enemies_obs{obs_space}_reward{reward_func}"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    
    return env

def train_pipeline(total_timesteps=50000):
    """Train models for all combinations of observation spaces and reward functions"""
    # Define configurations to test
    observation_spaces = [1, 2, 3]  # Only use observation spaces 1, 2, and 3
    reward_functions = [1, 2, 3]    # All reward functions
    
    # Create results storage
    results = []
    
    print("Starting training pipeline for maze with walls but no enemies")
    print(f"Training each model for {total_timesteps} timesteps")
    print("-" * 50)
    
    # Create output directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./tensorboard", exist_ok=True)
    
    # For each configuration
    for obs_space in observation_spaces:
        for reward_func in reward_functions:
            config_name = f"no_enemies_obs{obs_space}_reward{reward_func}"
            print(f"\n\nTraining configuration: {config_name}")
            
            # Create directories for this configuration
            model_dir = f"./models/{config_name}"
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(f"./logs/{config_name}", exist_ok=True)
            os.makedirs(f"./tensorboard/{config_name}", exist_ok=True)
            
            # Create environment
            env = create_custom_env(obs_space, reward_func)
            
            # Define policy kwargs for the feature extractor
            policy_kwargs = {
                "features_extractor_class": GridworldFeaturesExtractor,
                "features_extractor_kwargs": {"obs_space_type": obs_space, "features_dim": 64}
            }
            
            # Select appropriate policy based on observation space
            policy = "MultiInputPolicy" if obs_space == 3 else "MlpPolicy"
            
            # Define the model
            model = DQN(policy, env, verbose=1, 
                       learning_rate=0.0003, 
                       buffer_size=100000,
                       exploration_fraction=0.5,
                       exploration_final_eps=0.1,
                       policy_kwargs=policy_kwargs,
                       learning_starts=5000,
                       train_freq=4,
                       tensorboard_log=f"./tensorboard/{config_name}")
            
            # Setup evaluation callback
            eval_env = create_custom_env(obs_space, reward_func)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=model_dir,
                log_path=f"./logs/{config_name}",
                eval_freq=5000,
                n_eval_episodes=5,
                deterministic=True,
                render=False
            )
            
            # Train the model
            try:
                print(f"Starting training for {total_timesteps} timesteps...")
                model.learn(total_timesteps=total_timesteps, callback=eval_callback, 
                          tb_log_name=config_name, progress_bar=True)
                
                # Save the final model
                model_path = f"{model_dir}/final_model"
                model.save(model_path)
                print(f"Model saved to {model_path}")
                
                # Evaluate the model
                mean_reward, std_reward = evaluate_policy(
                    model, eval_env, n_eval_episodes=10, deterministic=True
                )
                
                # Record results
                results.append({
                    'Observation Space': obs_space,
                    'Reward Function': reward_func,
                    'Mean Reward': mean_reward,
                    'Std Reward': std_reward
                })
                
                print(f"Evaluation results: {mean_reward:.2f} ± {std_reward:.2f}")
                
            except Exception as e:
                print(f"Error training model: {e}")
                
                # Still record an entry with NaN values
                results.append({
                    'Observation Space': obs_space,
                    'Reward Function': reward_func,
                    'Mean Reward': float('nan'),
                    'Std Reward': float('nan')
                })
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv("no_enemies_results.csv", index=False)
    
    # Generate summary plot
    plot_results(results_df)
    
    print("\nTraining pipeline complete!")
    print(f"Results saved to no_enemies_results.csv")
    
    return results_df

def run_best_model():
    """Load and run the best model from the training pipeline"""
    if os.path.exists("no_enemies_results.csv"):
        results_df = pd.read_csv("no_enemies_results.csv")
        
        # Find the best model (highest mean reward)
        best_row = results_df.loc[results_df['Mean Reward'].idxmax()]
        best_obs = int(best_row['Observation Space'])
        best_reward = int(best_row['Reward Function'])
        
        print(f"Running best model: Observation Space {best_obs}, Reward Function {best_reward}")
        
        # Create custom environment with rendering
        custom.set_observation_space(best_obs)
        custom.set_active_reward_function(best_reward)
        
        env = gym.make("standard", 
                      render_mode="human", 
                      predefined_map=MAZE_NO_ENEMIES, 
                      activate_game_status=True)
        env = DangerAwareEnv(env)
        env = ObservationWrapper(env, obs_space_type=best_obs)
        
        # Load the model
        model_path = f"./models/no_enemies_obs{best_obs}_reward{best_reward}/final_model"
        if os.path.exists(f"{model_path}.zip"):
            model = DQN.load(model_path)
            
            # Run episodes
            for _ in range(3):
                obs, _ = env.reset()
                done = False
                total_reward = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                
                print(f"Episode finished with reward: {total_reward:.2f}")
                time.sleep(2)
            
            env.close()
        else:
            print(f"Model not found at {model_path}. Train models first.")
    else:
        print("No results file found. Run training pipeline first.")

def run_specific_combination(obs_space, reward_func):
    """Run a specific model from the training pipeline"""
    # Create custom environment with rendering
    custom.set_observation_space(obs_space)
    custom.set_active_reward_function(reward_func)
    
    env = gym.make("standard", 
                  render_mode="human", 
                  predefined_map=MAZE_NO_ENEMIES, 
                  activate_game_status=True)
    env = DangerAwareEnv(env)
    env = ObservationWrapper(env, obs_space_type=obs_space)
    
    # Load the model
    model_path = f"./models/no_enemies_obs{obs_space}_reward{reward_func}/final_model"
    if os.path.exists(f"{model_path}.zip"):
        model = DQN.load(model_path)
        
        # Run episodes
        for _ in range(3):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
            
            print(f"Episode finished with reward: {total_reward:.2f}")
            time.sleep(2)
        
        env.close()
    else:
        print(f"Model not found at {model_path}. Train models first.")

def plot_results(results_df):
    """Create a heatmap visualization of the training results"""
    if len(results_df) == 0:
        print("No results to plot")
        return
    
    # Create a pivot table for the heatmap
    pivot_data = results_df.pivot(
        index='Observation Space', 
        columns='Reward Function', 
        values='Mean Reward'
    )
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    im = plt.imshow(pivot_data, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, label='Mean Reward')
    
    # Add labels
    plt.title('Performance Comparison: Observation Spaces vs Reward Functions')
    plt.xlabel('Reward Function')
    plt.ylabel('Observation Space')
    
    # Set ticks
    plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
    plt.yticks(range(len(pivot_data.index)), pivot_data.index)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.iloc[i, j]
            if not np.isnan(value):
                plt.text(j, i, f'{value:.1f}', 
                        ha='center', va='center', 
                        color='white' if value < pivot_data.max().max() * 0.7 else 'black')
    
    # Save plot
    plt.tight_layout()
    plt.savefig('no_enemies_comparison.png')
    print("Results plot saved to no_enemies_comparison.png")
    
    # Display best combination
    best_obs = results_df.loc[results_df['Mean Reward'].idxmax()]['Observation Space']
    best_rew = results_df.loc[results_df['Mean Reward'].idxmax()]['Reward Function']
    best_score = results_df['Mean Reward'].max()
    
    print(f"\nBest combination: Observation Space {best_obs}, Reward Function {best_rew}")
    print(f"Best score: {best_score:.2f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            # Optional timesteps parameter
            timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
            train_pipeline(timesteps)
        elif sys.argv[1] == "run_best":
            run_best_model()
        elif sys.argv[1] == "run_specific" and len(sys.argv) >= 4:
            # Example: python maze_no_enemies.py run_specific 3 2
            obs_space = int(sys.argv[2])
            reward_func = int(sys.argv[3])
            run_specific_combination(obs_space, reward_func)
        else:
            print("Usage options:")
            print("  python maze_no_enemies.py train [timesteps]")
            print("  python maze_no_enemies.py run_best")
            print("  python maze_no_enemies.py run_specific <obs_space> <reward_func>")
    else:
        print("Usage options:")
        print("  python maze_no_enemies.py train [timesteps]")
        print("  python maze_no_enemies.py run_best")
        print("  python maze_no_enemies.py run_specific <obs_space> <reward_func>") 
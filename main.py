import os
import random
import sys
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import check_for_nested_spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn

# Add the 'coverage-gridworld' directory (the parent of the package folder) to sys.path
module_path = os.path.join(os.path.dirname(__file__), 'coverage-gridworld')
if module_path not in sys.path:
    sys.path.insert(0, module_path)

import coverage_gridworld  # must be imported, even though it's not directly referenced
from coverage_gridworld import custom

# Import environment registration
# from custom import observation_space
danger_table = [[[0 for _ in range(4)] for _ in range(10)] for _ in range(10)]
# New shit
def determine_cell_danger_tables(env, enemies):
    """
    Creates danger tables that predict where enemies will be looking in future timesteps.
    This helps the agent avoid being spotted by knowing which cells will be dangerous.
    """
    # Initialize danger tables for all 4 timesteps
    danger_tables = [[[0 for _ in range(4)] for _ in range(10)] for _ in range(10)]
    
    # Get the actual environment (unwrap if needed)
    actual_env = env
    while hasattr(actual_env, 'env'):
        actual_env = actual_env.env
    
    for e in enemies:
        # Start with current orientation
        orientation = e.orientation
        
        # For each future timestep (0=current, 1=next, etc)
        for timestep in range(4):
            # Check enemy's field of view based on orientation
            match orientation:
                case 0:  # LEFT
                    # Look up to 5 cells to the left (enemy can see 4 cells away)
                    for distance in range(1, 5):
                        x = e.x - distance
                        y = e.y
                        
                        # Check bounds
                        if x < 0:
                            break
                            
                        # Mark cell as dangerous for this timestep
                        danger_tables[x][y][timestep] = 1
                        
                        # Check if a wall blocks vision using FOV information
                        if not is_cell_visible(actual_env, y, x):
                            break
                            
                case 1:  # DOWN
                    for distance in range(1, 5):
                        x = e.x
                        y = e.y + distance
                        
                        if y > 9:
                            break
                            
                        danger_tables[x][y][timestep] = 1
                        
                        if not is_cell_visible(actual_env, y, x):
                            break
                            
                case 2:  # RIGHT
                    for distance in range(1, 5):
                        x = e.x + distance
                        y = e.y
                        
                        if x > 9:
                            break
                            
                        danger_tables[x][y][timestep] = 1
                        
                        if not is_cell_visible(actual_env, y, x):
                            break
                            
                case 3:  # UP
                    for distance in range(1, 5):
                        x = e.x
                        y = e.y - distance
                        
                        if y < 0:
                            break
                            
                        danger_tables[x][y][timestep] = 1
                        
                        if not is_cell_visible(actual_env, y, x):
                            break
            
            # Rotate orientation counter-clockwise for next timestep
            orientation = (orientation + 1) % 4
            
    return danger_tables

def is_cell_visible(env, i, j):
    """
    Check if a cell is visible (not blocked by walls or enemies).
    Reuses the environment's own visibility check.
    """
    # If the environment has this method, use it
    if hasattr(env, '_is_cell_visible'):
        return env._is_cell_visible(i, j)
    elif hasattr(env, '__is_cell_visible'):
        return env.__is_cell_visible(i, j)
    
    # Otherwise, implement simple bounds checking
    if i < 0 or j < 0 or i >= 10 or j >= 10:
        return False
    
    # Check directly if the cell is a wall or enemy - this is a fallback
    try:
        grid = env.grid
        if np.array_equal(grid[i, j], np.asarray((101, 67, 33))) or np.array_equal(grid[i, j], np.asarray((31, 198, 0))):
            return False
    except:
        # If we can't check directly, just assume it's visible
        pass
        
    return True

def human_player():
    # Write the letter for the desired movement in the terminal/console and then press Enter

    input_action = input()
    if input_action.lower() == "w":
        return 3
    elif input_action.lower() == "a":
        return 0
    elif input_action.lower() == "s":
        return 1
    elif input_action.lower() == "d":
        return 2
    elif input_action.isdigit():
        return int(input_action)
    else:
        return 4


def random_player():
    return random.randint(0, 4)

def rl_player(model, observation):
    action, _ = model.predict(observation, deterministic=True)
    return action


maps = [
    [
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
        [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
        [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
    ],
    [
        [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
        [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
        [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    ],
    [
        [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
    ],
    [
        [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 2, 0, 2, 0, 0, 2, 0, 2, 0]
    ]
]

# Custom environment wrapper to handle danger tables
class DangerAwareEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Initialize danger tables for enemy tracking
        self.timestep = 0
        self.danger_tables_initialized = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Take a no-op step to get the initial info
        no_op_action = 4  # STAY action
        _, _, _, _, initial_info = self.env.step(no_op_action)
        
        # Initialize danger tables based on enemy positions
        danger_tables = determine_cell_danger_tables(self.env, initial_info["enemies"])
        custom.set_danger_table(danger_tables)
        self.danger_tables_initialized = True
        custom.reset_timestep()
        self.timestep = 0
        
        return obs, info

    def step(self, action):
        # Increment timestep for tracking enemy rotation
        custom.incr_timestep()
        self.timestep = (self.timestep + 1) % 4
        
        # Take the step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update danger tables if game is not over
        if not terminated and not truncated and self.timestep % 4 == 0:
            # Periodically update danger tables to ensure they reflect current enemy positions
            danger_tables = determine_cell_danger_tables(self.env, info["enemies"])
            custom.set_danger_table(danger_tables)
            
        return obs, reward, terminated, truncated, info

# Custom feature extractor for handling different observation spaces
class GridworldFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, obs_space_type=1, features_dim=64):
        super().__init__(observation_space, features_dim)
        
        self.obs_space_type = obs_space_type
        
        if obs_space_type == 1 or obs_space_type == 2:
            # For flat MultiDiscrete spaces (full grid or local view)
            n_input = int(np.prod(observation_space.shape))
            self.extractor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_input, 128),
                nn.ReLU(),
                nn.Linear(128, features_dim),
                nn.ReLU(),
            )
        elif obs_space_type == 3:
            # For Dict space (feature-engineered)
            # Dynamic approach to handle variable input sizes
            self.net_agent_pos = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU()
            )
            
            self.net_danger = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU()
            )
            
            self.net_future_danger = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU()
            )
            
            self.net_progress = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU()
            )
            
            self.net_nearest = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU()
            )
            
            self.net_unexplored = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU()
            )
            
            self.net_walls = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU()
            )
            
            # Final layer to combine all features
            self.combiner = nn.Sequential(
                nn.Linear(72, features_dim),  # 16+8+8+8+16+8+8=72
                nn.ReLU()
            )
            
        elif obs_space_type == 4:
            # For Multi-layer observation
            # Reshape to handle the 2D grid with 2 channels
            n_input = int(np.prod(observation_space.shape))
            self.extractor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_input, 128),
                nn.ReLU(),
                nn.Linear(128, features_dim),
                nn.ReLU(),
            )

    def forward(self, observations):
        if self.obs_space_type == 1 or self.obs_space_type == 2 or self.obs_space_type == 4:
            # For flat spaces, just use the main extractor
            # Convert to float and normalize
            obs_float = observations.float() / 20.0  # Normalize with a safe maximum value
            return self.extractor(obs_float)
        else:
            # For Dict space, process each part separately and combine
            # More robust error handling for tensor shapes
            batch_size = observations["agent_position"].shape[0]
            device = observations["agent_position"].device
            
            # Create tensors with the right shapes based on what's available
            try:
                # Agent position (ensure it's exactly shape [batch_size, 2])
                if observations["agent_position"].shape[1] == 2:
                    agent_pos = observations["agent_position"].float() / 10.0  # Normalize
                else:
                    # Just take the first two dimensions if it's larger
                    agent_pos = observations["agent_position"][:, :2].float() / 10.0
                agent_features = self.net_agent_pos(agent_pos)
            except:
                # Fallback if there's an issue
                agent_features = th.zeros((batch_size, 16), device=device)
            
            # Danger
            try:
                danger_features = self.net_danger(observations["danger"].float())
            except:
                danger_features = th.zeros((batch_size, 8), device=device)
            
            # Future danger
            try:
                future_danger_features = self.net_future_danger(observations["future_danger"].float())
            except:
                future_danger_features = th.zeros((batch_size, 8), device=device)
            
            # Exploration progress
            try:
                progress_features = self.net_progress(observations["exploration_progress"])
            except:
                progress_features = th.zeros((batch_size, 8), device=device)
            
            # Nearest uncovered
            try:
                if observations["nearest_uncovered"].shape[1] == 2:
                    nearest = observations["nearest_uncovered"].float()
                    nearest_normalized = nearest.clone()
                    nearest_normalized[:, 0] = nearest[:, 0] / 20.0  # Distance
                    nearest_normalized[:, 1] = nearest[:, 1] / 4.0   # Direction
                    nearest_features = self.net_nearest(nearest_normalized)
                else:
                    # If shape is wrong, create a reasonable tensor
                    nearest_tensor = th.zeros((batch_size, 2), device=device)
                    nearest_features = self.net_nearest(nearest_tensor)
            except:
                nearest_features = th.zeros((batch_size, 16), device=device)
            
            # Unexplored direction
            try:
                if "unexplored_direction" in observations:
                    unexplored_features = self.net_unexplored(observations["unexplored_direction"].float())
                else:
                    unexplored_features = th.zeros((batch_size, 8), device=device)
            except:
                unexplored_features = th.zeros((batch_size, 8), device=device)
            
            # Wall blocks
            try:
                if "wall_blocks" in observations:
                    wall_features = self.net_walls(observations["wall_blocks"].float())
                else:
                    wall_features = th.zeros((batch_size, 8), device=device)
            except:
                wall_features = th.zeros((batch_size, 8), device=device)
            
            # Combine all features
            combined = th.cat([
                agent_features, 
                danger_features, 
                future_danger_features, 
                progress_features,
                nearest_features, 
                unexplored_features, 
                wall_features
            ], dim=1)
            
            return self.combiner(combined)


# Custom environment wrapper that properly handles observations for DQN
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, obs_space_type=1):
        super().__init__(env)
        self.obs_space_type = obs_space_type
        
        # Define appropriate observation space for each type
        if obs_space_type == 1:
            # Full grid - values 0-20 to account for future dangerous cells (+10)
            self.observation_space = gym.spaces.Box(
                low=0, high=20, shape=(100,), dtype=np.int8
            )
        elif obs_space_type == 2:
            # Local view - values 0-20
            self.observation_space = gym.spaces.Box(
                low=0, high=20, shape=(25,), dtype=np.int8
            )
        elif obs_space_type == 3:
            # Keep Dict space as is - Stable Baselines can handle it
            pass
        elif obs_space_type == 4:
            # Multi-layer - two layers stacked
            self.observation_space = gym.spaces.Box(
                low=0, high=10, shape=(200,), dtype=np.int8
            )

    def observation(self, obs):
        if self.obs_space_type == 3:
            # Dict space - return as is
            return obs
        else:
            # For array spaces, convert to numpy array
            return np.array(obs, dtype=np.int8)

def create_experiment_env(obs_space, reward_func, map_type="safe", render_mode=None, map_list=None):
    """Create and configure an environment with specific observation space and reward function"""
    # Set the observation space and reward function
    custom.set_observation_space(obs_space)
    custom.set_active_reward_function(reward_func)
    
    # Create the environment
    env = gym.make(map_type, render_mode=render_mode, predefined_map_list=map_list, activate_game_status=False)
    
    # Wrap with the danger aware wrapper
    env = DangerAwareEnv(env)
    
    # Wrap with observation wrapper
    env = ObservationWrapper(env, obs_space_type=obs_space)
    
    # Add monitoring for metrics
    log_dir = f"./logs/obs{obs_space}_reward{reward_func}"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    
    return env

def run_experiments():
    """
    Run experiments with different observation spaces and reward functions,
    collect metrics, and generate plots
    """
    # Parameters
    total_timesteps = 100000  # Adjust based on available time/resources
    eval_episodes = 10
    
    # Define configurations to test - all required combinations
    observation_spaces = [1, 2, 3, 4]  # All 4 observation spaces
    reward_functions = [1, 2, 3]  # All 3 reward functions
    map_types = ["safe", "maze", "chokepoint", "sneaky_enemies"]
    
    # Create results dataframe
    results = []
    
    # Training progress metrics
    training_metrics = {}
    
    # Create experiment directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./tensorboard", exist_ok=True)
    os.makedirs("./eval_logs", exist_ok=True)
    
    # For each configuration
    for obs_space in observation_spaces:
        for reward_func in reward_functions:
            config_name = f"obs{obs_space}_reward{reward_func}"
            print(f"\n\nTraining configuration: {config_name}")
            print("=" * 50)
            
            # Create environment
            env = create_experiment_env(obs_space, reward_func, "standard")  # Use random maps for training
            
            # Define model with custom policy kwargs for the feature extractor
            policy_kwargs = {
                "features_extractor_class": GridworldFeaturesExtractor,
                "features_extractor_kwargs": {"obs_space_type": obs_space, "features_dim": 64}
            }
            
            # Select the appropriate policy based on observation space
            policy = "MultiInputPolicy" if obs_space == 3 else "MlpPolicy"
            
            # Define the model with hyperparameters appropriate for this task
            model = DQN(policy, env, verbose=1, 
                      learning_rate=0.0003, 
                      buffer_size=50000,
                      exploration_fraction=0.2,
                      exploration_final_eps=0.05,
                      policy_kwargs=policy_kwargs,
                      learning_starts=1000,
                      target_update_interval=500,
                      tensorboard_log=f"./tensorboard/{config_name}")
            
            # Setup evaluation callback
            eval_log_dir = f"./eval_logs/{config_name}"
            os.makedirs(eval_log_dir, exist_ok=True)
            eval_callback = EvalCallback(
                create_experiment_env(obs_space, reward_func, "standard"),
                best_model_save_path=f"./models/{config_name}",
                log_path=eval_log_dir,
                eval_freq=10000,
                n_eval_episodes=5,
                deterministic=True,
                render=False
            )
            
            # Train the model
            try:
                model.learn(total_timesteps=total_timesteps, callback=eval_callback, 
                          tb_log_name=config_name, progress_bar=True)
                
                # Save the model
                model_path = f"./models/{config_name}/final_model"
                model.save(model_path)
                print(f"Model saved to {model_path}")
                
                # Extract training metrics from monitor
                monitor_path = f"./logs/obs{obs_space}_reward{reward_func}/monitor.csv"
                if os.path.exists(monitor_path):
                    training_data = pd.read_csv(monitor_path, skiprows=1)
                    training_metrics[config_name] = training_data
                    print(f"Training data extracted from {monitor_path}")
                else:
                    print(f"Warning: Monitor file not found at {monitor_path}")
                
                # Test on each map type
                for map_type in map_types:
                    print(f"Evaluating on {map_type}...")
                    
                    # Create test environment
                    test_env = create_experiment_env(obs_space, reward_func, map_type)
                    
                    # Evaluate policy
                    mean_reward, std_reward = evaluate_policy(
                        model, test_env, n_eval_episodes=eval_episodes, deterministic=True
                    )
                    
                    # Calculate exploration percentage and other metrics
                    exploration_percentages = []
                    step_counts = []
                    success_rate = 0
                    
                    for _ in range(eval_episodes):
                        obs, _ = test_env.reset()
                        done = False
                        steps = 0
                        success = False
                        
                        while not done:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, done, _, info = test_env.step(action)
                            steps += 1
                            
                            if done:
                                # Check if success (completed map without game over)
                                if not info["game_over"] and info["cells_remaining"] == 0:
                                    success = True
                                    success_rate += 1
                                
                                exploration_percentage = (info["total_covered_cells"] / info["coverable_cells"]) * 100
                                exploration_percentages.append(exploration_percentage)
                                step_counts.append(steps)
                    
                    # Calculate metrics
                    avg_exploration = np.mean(exploration_percentages)
                    std_exploration = np.std(exploration_percentages)
                    avg_steps = np.mean(step_counts)
                    success_rate = (success_rate / eval_episodes) * 100
                    
                    # Store results
                    results.append({
                        'Observation Space': f"Obs {obs_space}",
                        'Reward Function': f"Reward {reward_func}",
                        'Map Type': map_type,
                        'Mean Reward': mean_reward,
                        'Std Reward': std_reward,
                        'Exploration %': avg_exploration,
                        'Exploration Std': std_exploration,
                        'Avg Steps': avg_steps,
                        'Success Rate %': success_rate
                    })
                    
                    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
                    print(f"Exploration: {avg_exploration:.2f}% ± {std_exploration:.2f}%")
                    print(f"Success rate: {success_rate:.2f}%")
            
            except Exception as e:
                print(f"Error training model with {config_name}: {e}")
                # Still add an entry to results to show that we tried this configuration
                for map_type in map_types:
                    results.append({
                        'Observation Space': f"Obs {obs_space}",
                        'Reward Function': f"Reward {reward_func}",
                        'Map Type': map_type,
                        'Mean Reward': float('nan'),
                        'Std Reward': float('nan'),
                        'Exploration %': float('nan'),
                        'Exploration Std': float('nan'),
                        'Avg Steps': float('nan'),
                        'Success Rate %': float('nan')
                    })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv("experiment_results.csv", index=False)
    print("\nResults saved to experiment_results.csv")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(results_df, training_metrics)
    print("Plots saved to ./plots/ directory")
    
    # Print summary of best configurations
    print("\nBest configurations by map type:")
    best_configs = results_df.loc[results_df.groupby('Map Type')['Mean Reward'].idxmax()]
    print(best_configs[['Map Type', 'Observation Space', 'Reward Function', 'Mean Reward', 'Exploration %']])
    
    return results_df

def plot_results(results_df, training_metrics):
    """Generate plots to visualize the experimental results"""
    # Create plots directory
    os.makedirs("./plots", exist_ok=True)
    
    # Set the style
    plt.style.use('ggplot')
    sns.set(font_scale=1.2)
    
    # 1. Reward Comparison by Observation Space and Reward Function
    plt.figure(figsize=(16, 10))
    ax = sns.barplot(
        data=results_df, 
        x='Map Type', 
        y='Mean Reward', 
        hue='Observation Space',
        palette='Set2',
        errorbar=('ci', 95),
        capsize=0.1
    )
    ax.set_title('Mean Reward by Map Type and Observation Space', fontsize=18)
    ax.set_xlabel('Map Type', fontsize=14)
    ax.set_ylabel('Mean Reward', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/reward_by_obs_space.png', dpi=300)
    plt.close()
    
    # 2. Exploration Percentage Comparison
    plt.figure(figsize=(16, 10))
    ax = sns.barplot(
        data=results_df, 
        x='Map Type', 
        y='Exploration %', 
        hue='Reward Function',
        palette='Set3',
        errorbar=('ci', 95),
        capsize=0.1
    )
    ax.set_title('Exploration Percentage by Map Type and Reward Function', fontsize=18)
    ax.set_xlabel('Map Type', fontsize=14)
    ax.set_ylabel('Exploration %', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/exploration_by_reward.png', dpi=300)
    plt.close()
    
    # 3. Heatmap of mean rewards for each configuration
    pivot_df = results_df.pivot_table(
        index=['Observation Space', 'Reward Function'],
        columns='Map Type',
        values='Mean Reward'
    )
    
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        pivot_df, 
        annot=True, 
        cmap='viridis', 
        linewidths=0.5,
        fmt='.1f',
        cbar_kws={'label': 'Mean Reward'}
    )
    ax.set_title('Mean Reward Heatmap for All Configurations', fontsize=18)
    plt.tight_layout()
    plt.savefig('./plots/reward_heatmap.png', dpi=300)
    plt.close()
    
    # 4. Learning curves for each configuration
    plt.figure(figsize=(15, 10))
    
    for config, data in training_metrics.items():
        # Calculate rolling average of episode rewards
        if 'r' in data.columns:
            smoothed_rewards = data['r'].rolling(window=20).mean()
            plt.plot(data['l'].values, smoothed_rewards.values, label=config)
    
    plt.title('Learning Curves for Different Configurations', fontsize=18)
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Mean Episode Reward (smoothed)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./plots/learning_curves.png', dpi=300)
    plt.close()
    
    # 5. Observation space comparison for each reward function
    plt.figure(figsize=(16, 12))
    g = sns.catplot(
        data=results_df,
        x='Reward Function',
        y='Mean Reward',
        hue='Observation Space',
        col='Map Type',
        kind='bar',
        palette='deep',
        height=5,
        aspect=0.8,
        col_wrap=2,
        legend_out=False
    )
    g.set_axis_labels("Reward Function", "Mean Reward")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.savefig('./plots/obs_space_comparison.png', dpi=300)
    plt.close()
    
    # 6. Reward function comparison for each observation space
    plt.figure(figsize=(16, 12))
    g = sns.catplot(
        data=results_df,
        x='Observation Space',
        y='Exploration %',
        hue='Reward Function',
        col='Map Type',
        kind='bar',
        palette='Set3',
        height=5,
        aspect=0.8,
        col_wrap=2,
        legend_out=False
    )
    g.set_axis_labels("Observation Space", "Exploration %")
    g.set_titles("{col_name}")
    plt.tight_layout()
    plt.savefig('./plots/reward_func_comparison.png', dpi=300)
    plt.close()
    
    # 7. Radar chart of the best configuration performance
    try:
        # Find best configuration for each map type
        best_configs = results_df.loc[results_df.groupby('Map Type')['Mean Reward'].idxmax()]
        
        # Create radar chart
        categories = best_configs['Map Type'].tolist()
        N = len(categories)
        
        # Compute exploration percentages normalized to 0-100
        values = best_configs['Exploration %'].values
        
        # Create angles for each category
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        
        # Make the plot circular
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        categories = np.concatenate((categories, [categories[0]]))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Draw the chart
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Set category labels
        ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
        
        # Add labels for the best configuration
        for i, angle in enumerate(angles[:-1]):
            config = f"{best_configs.iloc[i]['Observation Space']}, {best_configs.iloc[i]['Reward Function']}"
            ax.text(angle, values[i] + 5, config, 
                    horizontalalignment='center', 
                    verticalalignment='center')
        
        plt.title('Best Configuration Performance Across Map Types', size=20)
        plt.tight_layout()
        plt.savefig('./plots/best_config_radar.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating radar chart: {e}")
    
    plt.close('all')


def demonstrate_best_model():
    """
    Load and demonstrate the best performing model
    """
    # Read results to find best model
    if os.path.exists("experiment_results.csv"):
        results_df = pd.read_csv("experiment_results.csv")
        
        # Find the best overall model (highest mean reward)
        best_row = results_df.loc[results_df['Mean Reward'].idxmax()]
        obs_space = int(best_row['Observation Space'].split()[1])
        reward_func = int(best_row['Reward Function'].split()[1])
        
        print(f"Demonstrating best model: Observation Space {obs_space}, Reward Function {reward_func}")
        
        # Configure environment
        custom.set_observation_space(obs_space)
        custom.set_active_reward_function(reward_func)
        
        # Load model
        model_path = f"./models/obs{obs_space}_reward{reward_func}/final_model"
        if os.path.exists(f"{model_path}.zip"):
            model = DQN.load(model_path)
            
            # Create demo environment with an easier map
            env = gym.make("safe", render_mode="human", predefined_map_list=None, activate_game_status=True)
            env = DangerAwareEnv(env)
            env = ObservationWrapper(env, obs_space_type=obs_space)
            
            # Run demo
            num_episodes = 3
            for i in range(num_episodes):
                obs, _ = env.reset()
                done = False
                while not done:
                    action = rl_player(model, obs)
                    obs, reward, done, truncated, info = env.step(action)
                if done:
                    time.sleep(2)
            env.close()
        else:
            print(f"Model not found at {model_path}. Run experiments first.")
    else:
        print("No experiment results found. Run experiments first.")

# Import scikit-learn components
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


class DQNEstimator(BaseEstimator):
    """
    A scikit-learn style wrapper for the Stable-Baselines3 DQN agent.
    This wrapper lets you use GridSearchCV for hyperparameter tuning.
    """
    def __init__(self, policy="MlpPolicy", learning_rate=0.001, gamma=0.99,
                 total_timesteps=1000, eval_episodes=3):
        self.policy = policy
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.total_timesteps = total_timesteps
        self.eval_episodes = eval_episodes

    def fit(self, X, y=None):
        # Create a new environment for training without rendering.

        # NOTE: Comment out whichever one you aren't using.
        # This one trains on random maps.
        self.env = gym.make(id="standard", render_mode="rgb_array", redefined_map_list=None, activate_game_status=False)
        # This one trains on the map set defined above.
        # self.env = gym.make(id="standard", render_mode="rgb_array", predefined_map_list=maps, activate_game_status=False)

        # Initialize the DQN model with the given hyperparameters.
        self.model = DQN(self.policy, self.env,
                         learning_rate=self.learning_rate,
                         gamma=self.gamma, verbose=0)
        self.model.set_random_seed(42)
        # Train the model.
        self.model.learn(total_timesteps=self.total_timesteps)
        # Save the model with a filename that encodes the hyperparameters.
        filename = f"dqn_{self.policy}_lr{self.learning_rate}_gamma{self.gamma}_timesteps{self.total_timesteps}.zip"
        self.model.save(filename)
        print(f"Model saved as {filename}")
        return self

    def predict(self, X):
        # For compatibility: given an observation X, predict the action.
        # Here, X is assumed to be a valid observation.
        action, _ = self.model.predict(X, deterministic=True)
        return action

    def score(self, X, y=None):
        # Evaluate the model by running a few episodes and computing the average reward.
        total_rewards = []
        self.model.save("temp.zip")
        self.model = DQN.load("temp.zip", env=gym.make("standard", render_mode="human", predefined_map_list=maps, activate_game_status=False))
        for _ in range(self.eval_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards)
        return avg_reward

def grid_search_hyperparameters():
    """
    Use scikit-learn's GridSearchCV to tune hyperparameters for the DQN agent.
    """
    # Define the hyperparameter grid.
    param_grid = {
        "learning_rate": [0.0001, 0.001, 0.01],
        "gamma": [0.95, 0.99]
    }
    # For demonstration, we use a smaller number of timesteps.
    estimator = DQNEstimator(total_timesteps=5000, eval_episodes=5)
    # Dummy data (not used by the estimator, but required by GridSearchCV)
    dummy_X = np.zeros((1, 1))
    # Create a single-fold CV by providing a list with one split.
    cv_split = [(np.array([0]), np.array([0]))]
    grid_search = GridSearchCV(estimator, param_grid, cv=cv_split)
    grid_search.fit(dummy_X)
    # Print detailed results from all trials
    print("\nDetailed Grid Search Results:")
    print("=" * 50)
    
    # Create a formatted table of results
    print(f"{'Parameters':<40} {'Mean Reward':<15} {'Rank':<10}")
    print("-" * 65)
    
    # Sort results by mean test score (descending)
    sorted_results = sorted(
        zip(grid_search.cv_results_['params'], 
            grid_search.cv_results_['mean_test_score'],
            grid_search.cv_results_['rank_test_score']),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Print each configuration with its performance
    for params, mean_score, rank in sorted_results:
        param_str = f"lr={params['learning_rate']}, gamma={params['gamma']}"
        print(f"{param_str:<40} {mean_score:<15.4f} {rank:<10}")
    
    # Print additional statistics if available
    if 'std_test_score' in grid_search.cv_results_:
        print("\nVariability in Results:")
        print(f"{'Parameters':<40} {'Mean Reward':<15} {'Std Dev':<15}")
        print("-" * 70)
        
        for i, params in enumerate(grid_search.cv_results_['params']):
            param_str = f"lr={params['learning_rate']}, gamma={params['gamma']}"
            mean = grid_search.cv_results_['mean_test_score'][i]
            std = grid_search.cv_results_['std_test_score'][i]
            print(f"{param_str:<40} {mean:<15.4f} {std:<15.4f}")

    # Additional statistics
    print("\nTiming Statistics:")
    print(f"{'Parameters':<40} {'Mean Fit Time':<15} {'Mean Score Time':<15}")
    print("-" * 70)
    
    for i, params in enumerate(grid_search.cv_results_['params']):
        param_str = f"lr={params['learning_rate']}, gamma={params['gamma']}"
        mean_fit_time = grid_search.cv_results_['mean_fit_time'][i]
        mean_score_time = grid_search.cv_results_['mean_score_time'][i]
        print(f"{param_str:<40} {mean_fit_time:<15.4f} {mean_score_time:<15.4f}")

    print("=" * 50)
    print("Best parameters found:", grid_search.best_params_)
    print("Best average reward:", grid_search.best_score_)
    model = DQN.load("dqn_MlpPolicy_lr{}_gamma{}_timesteps5000".format(grid_search.best_params_["learning_rate"], grid_search.best_params_["gamma"]))
    model.save("best_model_obs{}_reward{}.zip".format(custom.DEFAULT_OBSERVATION_SPACE, custom.DEFAULT_REWARD_FUNCTION))

def run_specific_model(obs_space=1, reward_func=1, map_type="safe"):
    """
    Load and run a specific model configuration on a specified map type
    """
    print(f"Running model with observation space {obs_space} and reward function {reward_func} on {map_type} map")
    
    # Configure environment
    custom.set_observation_space(obs_space)
    custom.set_active_reward_function(reward_func)
    
    # Create environment
    env = gym.make(map_type, render_mode="human", predefined_map_list=None, activate_game_status=True)
    env = DangerAwareEnv(env)
    env = ObservationWrapper(env, obs_space_type=obs_space)
    
    # Try to load the model
    model_path = f"./models/obs{obs_space}_reward{reward_func}/final_model"
    if os.path.exists(f"{model_path}.zip"):
        model = DQN.load(model_path)
        
        # Run for 3 episodes
        for _ in range(3):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            # Run episode
            while not done:
                action = rl_player(model, obs)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
            print(f"Episode finished with reward: {total_reward:.2f}")
            time.sleep(2)
        
        env.close()
    else:
        print(f"Model not found at {model_path}. Train this configuration first.")
        
    return


def train_specific_model(obs_space=3, reward_func=2, map_type="maze", total_timesteps=100000):
    """
    Train a single specific model configuration
    """
    print(f"Training model with observation space {obs_space} and reward function {reward_func}")
    
    # Configure environment
    custom.set_observation_space(obs_space)
    custom.set_active_reward_function(reward_func)
    
    # Create directories
    config_name = f"obs{obs_space}_reward{reward_func}"
    os.makedirs(f"./models/{config_name}", exist_ok=True)
    os.makedirs(f"./logs/{config_name}", exist_ok=True)
    os.makedirs(f"./tensorboard/{config_name}", exist_ok=True)
    
    # Create environment for training (using standard random maps)
    env = create_experiment_env(obs_space, reward_func, "standard")
    
    # Define policy kwargs for the feature extractor
    policy_kwargs = {
        "features_extractor_class": GridworldFeaturesExtractor,
        "features_extractor_kwargs": {"obs_space_type": obs_space, "features_dim": 64}
    }
    
    # Select appropriate policy based on observation space
    policy = "MultiInputPolicy" if obs_space == 3 else "MlpPolicy"
    
    # Define the model with improved exploration parameters
    model = DQN(policy, env, verbose=1, 
                learning_rate=0.0003, 
                buffer_size=100000,  # Doubled buffer size
                exploration_fraction=0.5,  # Explore more (was 0.3)
                exploration_final_eps=0.1,  # Higher final exploration rate (was 0.05)
                policy_kwargs=policy_kwargs,
                learning_starts=5000,  # Start learning after more observations
                train_freq=4,  # Update the model more frequently
                tensorboard_log=f"./tensorboard/{config_name}")
    
    # Setup evaluation callback
    eval_env = create_experiment_env(obs_space, reward_func, map_type)  # Use target map type for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{config_name}",
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
        model_path = f"./models/{config_name}/final_model"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Test on target map
        print(f"Evaluating on {map_type}...")
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return model
        
    except Exception as e:
        print(f"Error training model: {e}")
        return None


if __name__ == "__main__":
    # Allow choosing between running different modes via command-line argument
    if len(sys.argv) > 1:
        if sys.argv[1] == "grid_search":
            grid_search_hyperparameters()
        elif sys.argv[1] == "experiments":
            run_experiments()
        elif sys.argv[1] == "demo":
            demonstrate_best_model()
        elif sys.argv[1] == "run_model" and len(sys.argv) >= 4:
            # Format: python main.py run_model <obs_space> <reward_func> [map_type]
            obs_space = int(sys.argv[2])
            reward_func = int(sys.argv[3])
            map_type = sys.argv[4] if len(sys.argv) >= 5 else "safe"
            run_specific_model(obs_space, reward_func, map_type)
        elif sys.argv[1] == "train_model" and len(sys.argv) >= 4:
            # Format: python main.py train_model <obs_space> <reward_func> [map_type] [timesteps]
            obs_space = int(sys.argv[2])
            reward_func = int(sys.argv[3])
            map_type = sys.argv[4] if len(sys.argv) >= 5 else "maze"
            timesteps = int(sys.argv[5]) if len(sys.argv) >= 6 else 100000
            train_specific_model(obs_space, reward_func, map_type, timesteps)
        else:
            print("Usage options:")
            print("  python main.py grid_search")
            print("  python main.py experiments")
            print("  python main.py demo")
            print("  python main.py run_model <obs_space> <reward_func> [map_type]")
            print("  python main.py train_model <obs_space> <reward_func> [map_type] [timesteps]")
            print("\nExample: python main.py run_model 3 2 safe")
            print("Example: python main.py train_model 3 2 maze 100000")
    else:
        # Default behavior: simple demo with random actions
        env = gym.make("safe", render_mode="human", predefined_map_list=None, activate_game_status=True)
        
        num_episodes = 3
        for i in range(num_episodes):
            env.reset()
            obs, reward, done, truncated, info = env.step(4)
            danger_table = determine_cell_danger_tables(env, info["enemies"])
            custom.set_danger_table(danger_table)
            time.sleep(1)
            while not done:
                custom.incr_timestep()
                action = random_player()
                obs, reward, done, truncated, info = env.step(action)
            if done:
                time.sleep(2)
        env.close()

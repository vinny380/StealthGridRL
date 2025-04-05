import os
import random
import sys
import time
import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN

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
    # Take in location of enemies in particular
    danger_tables = [[[0 for _ in range(4)] for _ in range(10)] for _ in range(10)]
    for e in enemies:
        orientation = e.orientation
        # Try all 4 orientations counter clockwise
        for i in range(4):
            match orientation:
                case 0:
                    # Assume enemies can see 4 in any direction
                    for j in range(1, 5):
                        if e.x - j < 0: break
                        danger_tables[e.x - j][e.y][i] = 1
                case 1:
                    for j in range(1, 5):
                        if e.y + j > 9: break
                        danger_tables[e.x][e.y + j][i] = 1
                case 2:
                    for j in range(1, 5):
                        if e.x + j > 9: break
                        danger_tables[e.x + j][e.y][i] = 1
                case 3:
                    for j in range(1, 5):
                        if e.y - j < 0: break
                        danger_tables[e.x][e.y - j][i] = 1
            orientation = (orientation + 1) % 4
    return danger_tables

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

def rl_player():
    return random.randint(0, 4)


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

env = gym.make("safe", render_mode="human", predefined_map_list=None, activate_game_status=True)

num_episodes = 25

#model = DQN("MlpPolicy", env, verbose=0)
#model.learn(total_timesteps=1000)
#model.save("dqn")



def get_cell_danger_table():
    return danger_table

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
    estimator = DQNEstimator(total_timesteps=5000, eval_episodes=3)
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


if __name__ == "__main__":
    # Allow choosing between running the simulation or grid search via command-line argument.
    if len(sys.argv) > 1 and sys.argv[1] == "grid_search":
        grid_search_hyperparameters()
    else:
        for i in range(num_episodes):
            env.reset()
            obs, reward, done, truncated, info = env.step(4)
            danger_table = determine_cell_danger_tables(env, info["enemies"])
            custom.set_danger_table(danger_table)
            time.sleep(1)
            while not done:
                custom.incr_timestep()
                action = rl_player()
                obs, reward, done, truncated, info = env.step(action)
            if done:
                time.sleep(2)
        env.close()

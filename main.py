import random
import time
import gymnasium
import coverage_gridworld  # must be imported, even though it's not directly referenced
import numpy as np
import math  # For math.sqrt and other math functions
import matplotlib.pyplot as plt  # For plotting learning progress


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


class SafetyRewardWrapper:
    def __init__(self, env):
        self.env = env
        self.detection_penalty = 50.0
        self.safety_reward = 0.3
        self.proximity_penalty_factor = 0.5
        self.safe_exploration_bonus = 0.2
        self.exploration_reward = 1.0  # Increased reward for exploration
        self.grid_size = 10
        self.unexplored_cells = []
        self.last_min_distance = None
        self.visited_cells = set([(0, 0)])  # Start with agent's initial position as visited

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if done:
            return obs, reward, done, truncated, info

        # Extract information
        enemies = info["enemies"]
        agent_pos = info["agent_pos"]
        total_covered_cells = info["total_covered_cells"]
        coverable_cells = info["coverable_cells"]
        new_cell_covered = info["new_cell_covered"]
        
        # Calculate agent's coordinates
        agent_y, agent_x = agent_pos // self.grid_size, agent_pos % self.grid_size
        
        # Update visited cells set
        self.visited_cells.add((agent_y, agent_x))
        
        # Calculate custom reward
        custom_reward = 0.0
        
        # Strong reward for exploring new cells
        if new_cell_covered:
            custom_reward += self.exploration_reward
            
            # Bonus for exploring near walls (safer areas)
            wall_adjacent = False
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = agent_y + dy, agent_x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    if np.array_equal(self.env.unwrapped.grid[ny, nx], (101, 67, 33)):  # BROWN color for walls
                        wall_adjacent = True
                        break
            
            if wall_adjacent:
                custom_reward += self.safe_exploration_bonus
            
        # Calculate distances to nearest unexplored cells
        self.update_unexplored_cells(self.env.unwrapped.grid)
        distance_to_nearest = self.get_min_distance_to_unexplored(agent_y, agent_x)
        
        # Reward for moving toward unexplored cells
        if self.last_min_distance is not None and distance_to_nearest < self.last_min_distance:
            custom_reward += 0.2  # Reward for getting closer to unexplored areas
        
        # Save current distance for next step
        self.last_min_distance = distance_to_nearest
        
        # Analyze enemy positions and field of view
        fov_cells = []
        for enemy in enemies:
            fov_cells.extend(enemy.get_fov_cells())
        
        # Calculate minimum distance to any enemy FOV cell
        min_distance = float('inf')
        if fov_cells:
            for fov_y, fov_x in fov_cells:
                distance = ((fov_y - agent_y)**2 + (fov_x - agent_x)**2)**0.5
                min_distance = min(min_distance, distance)
        
        # Exponential penalty for proximity to danger
        if min_distance < 3:
            danger_factor = np.exp(3 - min_distance) - 1
            custom_reward -= self.proximity_penalty_factor * danger_factor

        # Return custom reward
        return obs, custom_reward, done, truncated, info
    
    def update_unexplored_cells(self, grid_3d):
        """Update the list of unexplored cells"""
        self.unexplored_cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if np.array_equal(grid_3d[i, j], (0, 0, 0)) or np.array_equal(grid_3d[i, j], (255, 0, 0)):
                    # BLACK or RED (unexplored or under enemy surveillance)
                    self.unexplored_cells.append((i, j))
    
    def get_min_distance_to_unexplored(self, agent_y, agent_x):
        """Calculate the minimum distance to any unexplored cell"""
        if not self.unexplored_cells:
            return 0
            
        min_distance = float('inf')
        for cell_y, cell_x in self.unexplored_cells:
            distance = ((cell_y - agent_y)**2 + (cell_x - agent_x)**2)**0.5
            min_distance = min(min_distance, distance)
        
        return min_distance
        
    def predict_enemy_fov_movement(self, enemies):
        """Predict the next position of enemy FOV cells based on their rotation pattern"""
        predicted_fov = []
        
        # Default FOV distance
        enemy_fov_distance = 4
        
        for enemy in enemies:
            # Get current orientation and predict next (counter-clockwise rotation)
            next_orientation = (enemy.orientation + 1) % 4
            
            # Get current position
            x, y = enemy.x, enemy.y
            
            # Predict FOV cells for next orientation
            for i in range(1, enemy_fov_distance + 1):
                if next_orientation == 0:  # LEFT
                    fov_y, fov_x = y, x - i
                elif next_orientation == 1:  # DOWN
                    fov_y, fov_x = y + i, x
                elif next_orientation == 2:  # RIGHT
                    fov_y, fov_x = y, x + i
                else:  # UP
                    fov_y, fov_x = y - i, x
                
                # Check if cell is valid
                if 0 <= fov_y < self.grid_size and 0 <= fov_x < self.grid_size:
                    # Cannot see through walls or other enemies
                    if (np.array_equal(self.env.unwrapped.grid[fov_y, fov_x], (101, 67, 33)) or 
                        np.array_equal(self.env.unwrapped.grid[fov_y, fov_x], (31, 198, 0))):
                        break
                    predicted_fov.append((fov_y, fov_x))
                else:
                    break  # Stop if boundary reached
                    
        return predicted_fov
        
    def reset(self, **kwargs):
        """Reset wrapper state when environment resets"""
        obs, info = self.env.reset(**kwargs)
        self.unexplored_cells = []
        self.last_min_distance = None
        self.visited_cells = set([(0, 0)])  # Reset visited cells
        return obs, info


class QLearningAgent:
    def __init__(self, n_actions=5, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.3, exploration_decay=0.995):
        self.n_actions = n_actions
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon
        self.exploration_decay = exploration_decay
        self.q_table = {}  # Dictionary to store state-action values
    
    def get_state_key(self, obs):
        """Convert observation to a hashable state representation"""
        # If observation is not a dictionary, we can't extract useful info
        if not isinstance(obs, dict) and not hasattr(obs, 'get'):
            return str(hash(str(obs)))
        
        # Extract agent position
        agent_pos = None
        if hasattr(obs, 'get'):
            agent_pos = obs.get('agent_position', None)
        
        # If we don't have agent position in the observation, try to get it from grid
        if agent_pos is None:
            # Look for a grey cell (agent) in the grid
            if isinstance(obs, np.ndarray):
                for i in range(len(obs)):
                    for j in range(len(obs[i])):
                        if np.array_equal(obs[i][j], np.array([160, 161, 161])):  # GREY color
                            agent_pos = (i, j)
                            break
        
        # If still no agent position, use a simplified state
        if agent_pos is None:
            return str(hash(str(obs)))
            
        # Identify nearby explored and unexplored cells
        # This helps the agent understand where it has been and where to go
        nearby_cells = []
        if isinstance(obs, np.ndarray):
            # Get dimensions
            grid_height, grid_width = obs.shape[0], obs.shape[1]
            
            # Check cells around the agent (in a 3x3 grid)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = agent_pos[0] + dy, agent_pos[1] + dx
                    
                    # Skip if outside grid
                    if ny < 0 or ny >= grid_height or nx < 0 or nx >= grid_width:
                        nearby_cells.append('X')  # 'X' represents out of bounds
                        continue
                    
                    # Check cell type by color
                    cell = obs[ny, nx]
                    if np.array_equal(cell, np.array([0, 0, 0])):
                        nearby_cells.append('U')  # Unexplored
                    elif np.array_equal(cell, np.array([255, 255, 255])):
                        nearby_cells.append('E')  # Explored
                    elif np.array_equal(cell, np.array([101, 67, 33])):
                        nearby_cells.append('W')  # Wall
                    elif np.array_equal(cell, np.array([31, 198, 0])):
                        nearby_cells.append('N')  # Enemy
                    elif np.array_equal(cell, np.array([255, 0, 0])) or np.array_equal(cell, np.array([255, 127, 127])):
                        nearby_cells.append('D')  # Danger (enemy FOV)
                    else:
                        nearby_cells.append('O')  # Other
        
        # Simplify state representation by only including important features
        # This reduces the size of the Q-table while keeping essential information
        state_tuple = (
            agent_pos,  # Agent position is crucial
            tuple(nearby_cells)  # Local environment around agent
        )
        
        return str(state_tuple)
    
    def choose_action(self, obs):
        """Select action using epsilon-greedy policy"""
        state_key = self.get_state_key(obs)
        
        # Exploration: random action
        if random.random() < self.exploration_rate:
            return random.randint(0, self.n_actions - 1)
        
        # Exploitation: best known action
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.n_actions
        
        return np.argmax(self.q_table[state_key])
    
    def learn(self, obs, action, reward, next_obs, done):
        """Update Q-values using the Q-learning update rule"""
        state_key = self.get_state_key(obs)
        next_state_key = self.get_state_key(next_obs)
        
        # Initialize Q-values if not already present
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.n_actions
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0] * self.n_actions
        
        # Q-learning update formula
        current_q = self.q_table[state_key][action]
        
        # Terminal state handling
        if done:
            max_future_q = 0
        else:
            max_future_q = max(self.q_table[next_state_key])
        
        # Calculate new Q-value
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        
        # Update Q-table
        self.q_table[state_key][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.01, self.exploration_rate)  # Minimum exploration
    
    def save_model(self, filename="q_learning_model.npy"):
        """Save Q-table to a file"""
        np.save(filename, self.q_table)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename="q_learning_model.npy"):
        """Load Q-table from a file"""
        try:
            self.q_table = np.load(filename, allow_pickle=True).item()
            print(f"Model loaded from {filename}")
        except:
            print(f"Could not load model from {filename}, using new Q-table")
            self.q_table = {}


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

# Create the base environment
base_env = gymnasium.make("sneaky_enemies", render_mode="human", predefined_map_list=maps, activate_game_status=True)

# Wrap the environment with our SafetyRewardWrapper
env = SafetyRewardWrapper(base_env)

# Create a Q-learning agent
agent = QLearningAgent(
    n_actions=5,
    learning_rate=0.1,
    discount_factor=0.99,
    exploration_rate=0.3,
    exploration_decay=0.995
)

# Try to load a pre-trained model if it exists
try:
    agent.load_model("q_learning_model.npy")
except:
    print("Starting with a new Q-learning model")

# Training parameters
num_episodes = 20  # Increased number of episodes for learning
total_rewards = []  # Track rewards across episodes
exploration_rates = []  # Track exploration rate decay
coverage_percents = []  # Track how much of the grid is covered

for i in range(num_episodes):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    step_count = 0
    max_coverage = 0  # Track maximum coverage in this episode
    
    while not done:
        # Choose action using the agent's policy
        action = agent.choose_action(obs)
        
        # Take the action
        next_obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1
        
        # Track coverage percentage
        if 'total_covered_cells' in info and 'coverable_cells' in info:
            coverage = info['total_covered_cells'] / info['coverable_cells']
            max_coverage = max(max_coverage, coverage)
        
        # Let the agent learn from this experience
        agent.learn(obs, action, reward, next_obs, done)
        
        # Update current observation
        obs = next_obs
        
        # Print info
        if step_count % 10 == 0:  # Print every 10 steps to reduce output
            print(f"Episode {i+1}, Step {step_count}, Reward: {reward:.2f}, Total: {episode_reward:.2f}, Epsilon: {agent.exploration_rate:.4f}")
        
        # Sleep to visualize the agent's actions
        time.sleep(0.05)  # Fast but still visible
        
        # Break if episode is too long
        if step_count >= 500:
            print("Episode too long, terminating")
            break
    
    # End of episode
    total_rewards.append(episode_reward)
    exploration_rates.append(agent.exploration_rate)
    coverage_percents.append(max_coverage * 100)  # Store as percentage
    
    print(f"Episode {i+1} finished - Steps: {step_count}, Total reward: {episode_reward:.2f}, Max Coverage: {max_coverage*100:.1f}%")
    
    # Save model periodically
    if (i+1) % 5 == 0:
        agent.save_model()
    
    time.sleep(1)  # Brief pause between episodes

# Save the trained model
agent.save_model()

# Print final stats
print("\nTraining completed!")
print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
print(f"Average coverage: {np.mean(coverage_percents):.1f}%")
print(f"Number of states in Q-table: {len(agent.q_table)}")

# Plot learning progress
plt.figure(figsize=(12, 8))

# Plot rewards
plt.subplot(3, 1, 1)
plt.plot(total_rewards, 'b-')
plt.title('Learning Progress')
plt.ylabel('Reward')
plt.grid(True)

# Plot exploration rate
plt.subplot(3, 1, 2)
plt.plot(exploration_rates, 'g-')
plt.ylabel('Exploration Rate')
plt.grid(True)

# Plot coverage percentage
plt.subplot(3, 1, 3)
plt.plot(coverage_percents, 'r-')
plt.xlabel('Episode')
plt.ylabel('Coverage %')
plt.grid(True)

plt.tight_layout()
plt.savefig('learning_progress.png')
plt.show()

env.close()

def demonstrate_agent(model_path="q_learning_model.npy", episodes=3, render_mode="human"):
    """
    Demonstrate a trained agent's performance with minimal exploration
    """
    # Create environment for demonstration
    demo_env = gymnasium.make("sneaky_enemies", render_mode=render_mode, predefined_map_list=maps, activate_game_status=True)
    demo_env = SafetyRewardWrapper(demo_env)
    
    # Create agent with minimal exploration
    demo_agent = QLearningAgent(exploration_rate=0.05)  # Still some exploration to avoid getting stuck
    
    # Load the trained model
    try:
        demo_agent.load_model(model_path)
        print(f"Demonstrating agent loaded from {model_path}")
    except:
        print(f"Could not load model from {model_path}, demonstration may not be effective")
    
    total_steps = 0
    success_count = 0
    
    for i in range(episodes):
        obs, info = demo_env.reset()
        done = False
        episode_steps = 0
        episode_reward = 0
        
        print(f"\nStarting demonstration episode {i+1}")
        
        while not done:
            # Choose action using the agent's policy (mostly exploitation)
            action = demo_agent.choose_action(obs)
            
            # Take the action
            obs, reward, done, truncated, info = demo_env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Print coverage info
            if episode_steps % 20 == 0:
                if 'total_covered_cells' in info and 'coverable_cells' in info:
                    coverage = info['total_covered_cells'] / info['coverable_cells'] * 100
                    print(f"Step {episode_steps}, Coverage: {coverage:.1f}%")
            
            # Sleep to visualize
            time.sleep(0.1)
            
            # Break if episode is too long
            if episode_steps >= 500:
                print("Episode too long, terminating")
                break
        
        # Check success condition
        if 'total_covered_cells' in info and 'coverable_cells' in info:
            final_coverage = info['total_covered_cells'] / info['coverable_cells'] * 100
            print(f"Episode {i+1} finished - Steps: {episode_steps}, Reward: {episode_reward:.2f}, Coverage: {final_coverage:.1f}%")
            
            # Count as success if coverage is high
            if final_coverage > 80:
                success_count += 1
        else:
            print(f"Episode {i+1} finished - Steps: {episode_steps}, Reward: {episode_reward:.2f}")
        
        total_steps += episode_steps
        time.sleep(1)  # Pause between episodes
    
    demo_env.close()
    
    # Print summary
    print(f"\nDemonstration completed over {episodes} episodes")
    print(f"Average steps per episode: {total_steps/episodes:.1f}")
    print(f"Success rate: {success_count/episodes*100:.1f}%")

# Run a demonstration after training
demonstrate_agent()

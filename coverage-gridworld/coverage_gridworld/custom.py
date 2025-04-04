import numpy as np
import gymnasium as gym
from env import *

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

# Store danger table here within custom.py
danger_table = [[[0 for _ in range(4)] for _ in range(10)] for _ in range(10)]
danger_table_set = False

# Active reward function (1, 2, or 3)
# 1 = Basic exploration reward
# 2 = Predictive danger avoidance reward
# 3 = Distance-based exploration reward
# Default value used if no reward function is specified
DEFAULT_REWARD_FUNCTION = 1

# New shit
def determine_cell_danger_tables(enemies):
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

def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
    """
    # The grid has (10, 10, 3) shape and can store values from 0 to 255 (uint8). To use the whole grid as the
    # observation space, we can consider a MultiDiscrete space with values in the range [0, 256).

    # Make a grid with the symbolic representation, rather than color.
    cell_values = np.zeros(shape=(10, 10), dtype=np.uint8) + len(COLOR_IDS)
    # if MultiDiscrete is used, it's important to flatten() numpy arrays!
    return gym.spaces.MultiDiscrete(cell_values.flatten())


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    # If the observation returned is not the same shape as the observation_space, an error will occur!
    # Make sure to make changes to both functions accordingly.

    # If danger table has not been added, do so
    
    cell_values = np.ndarray(shape=(10, 10), dtype=np.uint8)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            symbol = tuple(grid[x][y])
            cell_values[x][y] = list(COLOR_IDS.keys())[list(COLOR_IDS.values()).index(symbol)]
    return cell_values.flatten()


def reward_function_1(info: dict) -> float:
    """
    Basic Exploration Reward Function
    
    This reward function focuses on efficient exploration with simple penalties:
    1. Rewards exploring new cells with a bonus relative to the difficulty (fewer cells = higher reward)
    2. Small penalty for each step to encourage efficiency
    3. Large penalty for being spotted by enemies
    4. Bonus reward for completing the level
    """
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    
    reward = 0
    
    # Penalty for each step to encourage efficiency
    reward -= 1
    
    # Reward for covering a new cell (higher reward for more difficult maps with fewer coverable cells)
    if new_cell_covered:
        reward += max(50, 200 - coverable_cells)
    
    # Large penalty for game over (being spotted)
    if game_over:
        reward -= 500
    
    # Bonus for completing the level
    if cells_remaining == 0:
        reward += 1000
        
    return reward


def reward_function_2(info: dict) -> float:
    """
    Predictive Danger Avoidance Reward
    
    This reward function uses the danger table to predict future danger and encourages the agent to:
    1. Visit cells that will become dangerous in the future
    2. Strongly rewards exploring new cells
    3. Penalizes getting caught
    4. Gives a completion bonus
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    
    # Set up danger table if not already done
    global danger_table_set
    if not danger_table_set:
        set_danger_table(determine_cell_danger_tables(enemies))
    
    reward = 0
    
    # Base step penalty
    reward -= 2
    
    # Major reward for exploring new cells
    if new_cell_covered:
        reward += 100
    
    # Severe penalty for getting caught
    if game_over:
        reward -= 1000
        reset_timestep()
        danger_table_set = False
        return reward
    
    # Completion bonus
    if cells_remaining == 0:
        reward += 2000
    
    # Predictive danger rewards - visit cells that will be dangerous soon
    cur_timestep = get_timestep()
    next_steps = [(cur_timestep + 1) % 4, (cur_timestep + 2) % 4, (cur_timestep + 3) % 4]
    
    # Convert agent_pos to x,y coordinates
    agent_x = agent_pos % 10
    agent_y = agent_pos // 10
    
    # Add reward for being in cells that will become dangerous soon
    for index, step in enumerate(next_steps):
        if danger_table[agent_x][agent_y][step] == 1:
            # Higher reward for cells that will be dangerous sooner
            reward += 10 * (3 - index)
    
    # Update timestep for tracking enemy rotation
    incr_timestep()
    
    return reward


def reward_function_3(info: dict) -> float:
    """
    Distance-Based Exploration Reward
    
    This reward function encourages:
    1. Exploring new cells with diminishing returns as more are explored
    2. Proximity to unexplored cells
    3. Avoiding getting caught
    4. Completion
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    
    reward = 0
    
    # Small step penalty
    reward -= 1
    
    # Reward for new cells with diminishing returns
    # As more cells are covered, the reward per cell decreases
    if new_cell_covered:
        # Calculate what percentage of cells have been explored
        exploration_progress = total_covered_cells / coverable_cells
        # Higher reward for early exploration, lower for later cells
        reward += max(20, 150 * (1 - exploration_progress))
    
    # Harsh penalty for getting caught
    if game_over:
        reward -= 500
        return reward
    
    # Large completion bonus with time efficiency factor
    if cells_remaining == 0:
        time_efficiency_bonus = steps_remaining / 500.0  # Higher bonus for faster completion
        reward += 1000 + (1000 * time_efficiency_bonus)
    
    # Convert agent_pos to x,y coordinates
    agent_x = agent_pos % 10
    agent_y = agent_pos // 10
    
    # Reward proximity to unexplored cells (if there are any left)
    if cells_remaining > 0:
        # This would require access to the grid state to calculate
        # Since we don't have direct access, we'll use a proxy reward based on cells_remaining
        exploration_urgency = cells_remaining / coverable_cells
        reward += 5 * exploration_urgency
    
    return reward


def reward(info: dict, active_reward_function: int = DEFAULT_REWARD_FUNCTION) -> float:
    """
    Main reward function that calls the specified reward function based on active_reward_function parameter.
    
    Args:
        info: The information dictionary from the environment
        active_reward_function: Which reward function to use (1, 2, or 3)
        
    Returns:
        float: The calculated reward
    """
    if active_reward_function == 1:
        return reward_function_1(info)
    elif active_reward_function == 2:
        return reward_function_2(info)
    elif active_reward_function == 3:
        return reward_function_3(info)
    else:
        # Default to reward function 1 if invalid selection
        return reward_function_1(info)


timestep = 0

def get_timestep():
    global timestep
    return timestep

def incr_timestep():
    global timestep
    timestep += 1

def reset_timestep():
    global timestep
    timestep = 0

def set_danger_table(new_table):
    global danger_table
    danger_table = new_table
    global danger_table_set
    danger_table_set = True

def set_active_reward_function(function_number):
    """
    Set the default reward function to use (1, 2, or 3)
    """
    global DEFAULT_REWARD_FUNCTION
    if function_number in [1, 2, 3]:
        DEFAULT_REWARD_FUNCTION = function_number
        print(f"Using reward function {function_number} as default")
    else:
        print(f"Invalid reward function number: {function_number}. Using default (1)")
        DEFAULT_REWARD_FUNCTION = 1
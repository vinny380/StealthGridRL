import numpy as np
import gymnasium as gym

# action IDs
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4

# rendering colors
BLACK = (0, 0, 0)            # unexplored cell
WHITE = (255, 255, 255)      # explored cell
BROWN = (101, 67, 33)        # wall
GREY = (160, 161, 161)       # agent
GREEN = (31, 198, 0)         # enemy
RED = (255, 0, 0)            # unexplored cell being observed by an enemy
LIGHT_RED = (255, 127, 127)  # explored cell being observed by an enemy

# color IDs
COLOR_IDS = {
    0: BLACK,      # unexplored cell
    1: WHITE,      # explored cell
    2: BROWN,      # wall
    3: GREY,       # agent
    4: GREEN,      # enemy
    5: RED,        # unexplored cell being observed by an enemy
    6: LIGHT_RED,  # explored cell being observed by an enemy
}

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

# Active observation space (1, 2, 3, or 4)
# 1 = Full grid observation (original)
# 2 = Local view observation (agent-centered 5x5 window)
# 3 = Feature-engineered observation (compact representation)
# 4 = Multi-layer observation (cell type and danger values)
DEFAULT_OBSERVATION_SPACE = 1

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

def observation_space_1(env: gym.Env) -> gym.spaces.Space:
    """
    Original full grid observation space.
    Returns the entire grid as a flattened MultiDiscrete space.
    """
    # The grid has (10, 10, 3) shape and can store values from 0 to 255 (uint8). To use the whole grid as the
    # observation space, we can consider a MultiDiscrete space with values in the range [0, 256).

    # Make a grid with the symbolic representation, rather than color.
    cell_values = np.zeros(shape=(10, 10), dtype=np.uint8) + len(COLOR_IDS)
    # if MultiDiscrete is used, it's important to flatten() numpy arrays!
    return gym.spaces.MultiDiscrete(cell_values.flatten())


def observation_space_2(env: gym.Env) -> gym.spaces.Space:
    """
    Local view observation space.
    Returns a 5x5 window centered on the agent, representing the local surroundings.
    This is more efficient for learning since it reduces the state space and provides
    a locality-focused representation that generalizes better across different maps.
    
    The space includes:
    - Cell types in a 5x5 window (agent is always at center)
    - Values represent the cell type according to COLOR_IDS
    """
    # 5x5 grid of cell types (values 0-6)
    local_view = np.zeros(shape=(5, 5), dtype=np.uint8) + len(COLOR_IDS)
    return gym.spaces.MultiDiscrete(local_view.flatten())


def observation_space_3(env: gym.Env) -> gym.spaces.Space:
    """
    Feature-engineered compact observation space.
    
    Enhanced version with:
    1. Better tracking of unexplored areas
    2. Multiple nearest unexplored cells (not just one)
    3. Improved directional guidance
    4. Wall block information
    """
    # Create a Dict space with named features
    return gym.spaces.Dict({
        # Agent position (x, y coordinates)
        'agent_position': gym.spaces.MultiDiscrete([10, 10]),
        
        # Danger levels in each direction (immediate danger)
        'danger': gym.spaces.MultiBinary(4),
        
        # Future danger levels (will this cell be dangerous in next 1-4 steps)
        'future_danger': gym.spaces.MultiBinary(4),
        
        # Exploration progress 
        'exploration_progress': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        
        # Direction to nearest uncovered cell
        'nearest_uncovered': gym.spaces.MultiDiscrete([20, 4]),  # (distance, direction)
        
        # Information about unexplored areas in each direction
        'unexplored_direction': gym.spaces.MultiBinary(4),
        
        # Information about walls blocking paths
        'wall_blocks': gym.spaces.MultiBinary(4)
    })


def observation_space_4(env: gym.Env) -> gym.spaces.Space:
    """
    Multi-layer observation space.
    
    This observation space uses two separate layers:
    - Layer 1: Cell types (agent, covered, uncovered, untraversable)
    - Layer 2: Danger values (representing enemy field of view and future danger)
    
    This provides a richer spatial representation of the environment.
    """
    # Cell type layer: 0=unexplored, 1=explored, 2=wall/enemy, 3=agent
    cell_values = np.zeros(shape=(10, 10), dtype=np.uint8) + 4
    
    # Danger layer: values 0-4 representing danger levels
    danger_values = np.zeros(shape=(10, 10), dtype=np.uint8) + 5
    
    # Stack the layers and flatten
    cell_values = np.stack((cell_values, danger_values), axis=-1)
    return gym.spaces.MultiDiscrete(cell_values.flatten())


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Main observation space function that selects the active observation space.
    """
    if DEFAULT_OBSERVATION_SPACE == 1:
        return observation_space_1(env)
    elif DEFAULT_OBSERVATION_SPACE == 2:
        return observation_space_2(env)
    elif DEFAULT_OBSERVATION_SPACE == 3:
        return observation_space_3(env)
    elif DEFAULT_OBSERVATION_SPACE == 4:
        return observation_space_4(env)
    else:
        # Default to observation space 1 if invalid selection
        return observation_space_1(env)


def observation_1(grid: np.ndarray):
    """
    Original full grid observation function.
    """
    # First, process the grid normally to get cell types
    cell_values = np.ndarray(shape=(10, 10), dtype=np.uint8)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            symbol = tuple(grid[x][y])
            cell_values[x][y] = list(COLOR_IDS.keys())[list(COLOR_IDS.values()).index(symbol)]
    
    # Now, if danger table is set, modify the cell values to indicate future danger
    if danger_table_set:
        cur_timestep = get_timestep()
        next_timestep = (cur_timestep + 1) % 4  # Look one step ahead
        
        # Create a special value (7) to indicate cells that will be dangerous in the next step
        # but aren't currently being observed
        for x in range(10):
            for y in range(10):
                # Only modify if the cell is not already marked as dangerous (5 or 6)
                if cell_values[x][y] not in [5, 6]:
                    if danger_table[x][y][next_timestep] == 1:
                        # Add 10 to differentiate regular cells from future-dangerous cells
                        # We'll decode this in the learning algorithm
                        cell_values[x][y] += 10
    
    return cell_values.flatten()


def observation_2(grid: np.ndarray):
    """
    Local view observation function.
    Returns a 5x5 window centered on the agent.
    """
    # Find agent position
    agent_found = False
    agent_x, agent_y = 0, 0
    
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if tuple(grid[x][y]) == GREY:
                agent_x, agent_y = x, y
                agent_found = True
                break
        if agent_found:
            break
    
    # Create 5x5 local view centered on agent
    local_view = np.zeros((5, 5), dtype=np.uint8)
    
    for i in range(5):
        for j in range(5):
            # Calculate corresponding position in the full grid
            grid_x = agent_x + (i - 2)
            grid_y = agent_y + (j - 2)
            
            # Check if the position is within grid bounds
            if 0 <= grid_x < 10 and 0 <= grid_y < 10:
                symbol = tuple(grid[grid_x][grid_y])
                cell_type = list(COLOR_IDS.keys())[list(COLOR_IDS.values()).index(symbol)]
                
                # Check for future danger if danger table is set
                if danger_table_set and cell_type not in [5, 6]:
                    cur_timestep = get_timestep()
                    next_timestep = (cur_timestep + 1) % 4  # Look one step ahead
                    
                    if danger_table[grid_x][grid_y][next_timestep] == 1:
                        # Add 10 to cell type to indicate future danger
                        cell_type += 10
                
                local_view[i, j] = cell_type
            else:
                # Out of bounds - mark as wall
                local_view[i, j] = 2  # BROWN (wall)
    
    return local_view.flatten()


def observation_3(grid: np.ndarray):
    """
    Feature-engineered compact observation function.
    
    Enhanced version with:
    1. Better tracking of unexplored areas
    2. Multiple nearest unexplored cells (not just one)
    3. Improved directional guidance
    """
    # Find agent position
    agent_found = False
    agent_x, agent_y = 0, 0
    
    # Count of total cells and explored cells
    total_cells = 0
    explored_cells = 0
    
    # Create a map of the grid with cell types
    cell_types = np.zeros((10, 10), dtype=np.uint8)
    
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            symbol = tuple(grid[x][y])
            cell_type = list(COLOR_IDS.keys())[list(COLOR_IDS.values()).index(symbol)]
            cell_types[x, y] = cell_type
            
            # Find agent
            if cell_type == 3:  # GREY (agent)
                agent_x, agent_y = x, y
                agent_found = True
            
            # Count cells that aren't walls
            if cell_type != 2:  # Not a wall
                total_cells += 1
                # Count explored cells
                if cell_type == 1 or cell_type == 3 or cell_type == 6:  # WHITE, GREY, LIGHT_RED
                    explored_cells += 1
    
    # Calculate exploration progress
    exploration_progress = explored_cells / max(1, total_cells)
    
    # Track multiple nearby unexplored cells (collect the 3 nearest ones)
    unexplored_cells = []
    
    for x in range(10):
        for y in range(10):
            if cell_types[x, y] == 0 or cell_types[x, y] == 5:  # BLACK or RED (unexplored)
                dist = abs(x - agent_x) + abs(y - agent_y)  # Manhattan distance
                
                # Store this unexplored cell info
                if dist < 20:  # Only consider cells within a reasonable distance
                    # Determine primary direction
                    if abs(x - agent_x) > abs(y - agent_y):
                        dir_x = -1 if x < agent_x else 1
                        dir_y = 0
                    else:
                        dir_x = 0
                        dir_y = -1 if y < agent_y else 1
                    
                    unexplored_cells.append((dist, x, y, dir_x, dir_y))
    
    # Sort by distance and take the 3 nearest cells
    unexplored_cells.sort()
    unexplored_cells = unexplored_cells[:3]
    
    # If we found any unexplored cells, use the nearest one
    if unexplored_cells:
        nearest_dist, nearest_x, nearest_y, dir_x, dir_y = unexplored_cells[0]
        
        # Get primary direction as a single number (0=left, 1=down, 2=right, 3=up)
        if dir_x < 0:
            nearest_dir = 0  # LEFT
        elif dir_y > 0:
            nearest_dir = 1  # DOWN
        elif dir_x > 0:
            nearest_dir = 2  # RIGHT
        else:
            nearest_dir = 3  # UP
    else:
        # If no unexplored cells found, use defaults
        nearest_dist = 0
        nearest_dir = 0
        
    # Calculate directions to unexplored areas (scan in 4 directions)
    unexplored_direction = [0, 0, 0, 0]  # LEFT, DOWN, RIGHT, UP
    
    # Check for unexplored cells in each direction
    # LEFT direction
    for x in range(agent_x - 1, -1, -1):
        if cell_types[x, agent_y] == 2:  # Stop at walls
            break
        if cell_types[x, agent_y] == 0 or cell_types[x, agent_y] == 5:
            unexplored_direction[0] = 1
            break
    
    # DOWN direction
    for y in range(agent_y + 1, 10):
        if cell_types[agent_x, y] == 2:  # Stop at walls
            break
        if cell_types[agent_x, y] == 0 or cell_types[agent_x, y] == 5:
            unexplored_direction[1] = 1
            break
    
    # RIGHT direction
    for x in range(agent_x + 1, 10):
        if cell_types[x, agent_y] == 2:  # Stop at walls
            break
        if cell_types[x, agent_y] == 0 or cell_types[x, agent_y] == 5:
            unexplored_direction[2] = 1
            break
    
    # UP direction
    for y in range(agent_y - 1, -1, -1):
        if cell_types[agent_x, y] == 2:  # Stop at walls
            break
        if cell_types[agent_x, y] == 0 or cell_types[agent_x, y] == 5:
            unexplored_direction[3] = 1
            break
    
    # Check immediate danger in each direction
    danger = [0, 0, 0, 0]
    # Left
    if agent_x > 0 and (cell_types[agent_x-1, agent_y] == 5 or cell_types[agent_x-1, agent_y] == 6):
        danger[0] = 1
    # Down
    if agent_y < 9 and (cell_types[agent_x, agent_y+1] == 5 or cell_types[agent_x, agent_y+1] == 6):
        danger[1] = 1
    # Right
    if agent_x < 9 and (cell_types[agent_x+1, agent_y] == 5 or cell_types[agent_x+1, agent_y] == 6):
        danger[2] = 1
    # Up
    if agent_y > 0 and (cell_types[agent_x, agent_y-1] == 5 or cell_types[agent_x, agent_y-1] == 6):
        danger[3] = 1
    
    # Get future danger from danger table
    future_danger = [0, 0, 0, 0]
    if danger_table_set:
        cur_timestep = get_timestep()
        for i in range(4):
            future_timestep = (cur_timestep + i) % 4
            future_danger[i] = danger_table[agent_x][agent_y][future_timestep]
    
    # Also check if paths in each direction are blocked by walls
    wall_blocks = [0, 0, 0, 0]  # LEFT, DOWN, RIGHT, UP
    if agent_x > 0 and cell_types[agent_x-1, agent_y] == 2:
        wall_blocks[0] = 1
    if agent_y < 9 and cell_types[agent_x, agent_y+1] == 2:
        wall_blocks[1] = 1
    if agent_x < 9 and cell_types[agent_x+1, agent_y] == 2:
        wall_blocks[2] = 1
    if agent_y > 0 and cell_types[agent_x, agent_y-1] == 2:
        wall_blocks[3] = 1
    
    # Build and return the feature dictionary
    return {
        'agent_position': np.array([agent_x, agent_y]),
        'danger': np.array(danger),
        'future_danger': np.array(future_danger),
        'exploration_progress': np.array([exploration_progress], dtype=np.float32),
        'nearest_uncovered': np.array([nearest_dist, nearest_dir]),
        'unexplored_direction': np.array(unexplored_direction),
        'wall_blocks': np.array(wall_blocks)
    }


def observation_4(grid: np.ndarray):
    """
    Multi-layer observation function (from custom_2.py).
    Returns a stacked representation with cell type and danger values.
    """
    cell_values = np.zeros(shape=(10, 10), dtype=np.uint8)
    danger_values = np.zeros(shape=(10, 10), dtype=np.uint8)

    enemies = []
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            symbol = tuple(grid[x][y])
            symbol_num = list(COLOR_IDS.keys())[list(COLOR_IDS.values()).index(symbol)]
            # Convert symbol numbers to simplified cell types
            match symbol_num:
                case 0, 5:  # BLACK, RED (unexplored)
                    cell_values[x][y] = 0
                case 1, 6:  # WHITE, LIGHT_RED (explored)
                    cell_values[x][y] = 1
                case 2, 4:  # BROWN, GREEN (wall, enemy)
                    cell_values[x][y] = 2
                case 3:     # GREY (agent)
                    cell_values[x][y] = 3
            
            # Track enemies for FOV calculation
            if symbol == GREEN:
                enemies.append((x, y))
            
            # Mark cells currently under observation
            if symbol == RED or symbol == LIGHT_RED:
                danger_values[x][y] = 4

    # Calculate danger values from enemies' fields of view
    for (ex, ey) in enemies:
        # Check in the four cardinal directions
        for (x_, y_) in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
            # Look up to 4 cells in each direction
            for r in range(1, 5):
                cx, cy = (ex + r*x_, ey + r*y_)
                # Make sure the cell is in bounds
                if cx not in range(10) or cy not in range(10):
                    break
                # Check if vision is obstructed
                if tuple(grid[cx][cy]) == BROWN or tuple(grid[cx][cy]) == GREEN:
                    break
                # Set danger value (add and take mod 5 to account for multiple enemies)
                danger_values[cx][cy] = (danger_values[cx][cy] + 1) % 5

    # Stack and flatten the two layers
    cell_values = np.stack((cell_values, danger_values), axis=-1)
    return cell_values.flatten()


def observation(grid: np.ndarray):
    """
    Main observation function that selects the active observation implementation.
    """
    if DEFAULT_OBSERVATION_SPACE == 1:
        return observation_1(grid)
    elif DEFAULT_OBSERVATION_SPACE == 2:
        return observation_2(grid)
    elif DEFAULT_OBSERVATION_SPACE == 3:
        return observation_3(grid)
    elif DEFAULT_OBSERVATION_SPACE == 4:
        return observation_4(grid)
    else:
        # Default to observation 1 if invalid selection
        return observation_1(grid)


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
    5. Encourages finding unexplored regions
    6. Prevents the agent from getting stuck in one area
    7. Adds special handling for known problem areas (10,3)
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]  # Use this to encourage efficiency
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    
    # Set up danger table if not already done
    global danger_table_set
    if not danger_table_set:
        set_danger_table(determine_cell_danger_tables(enemies))
    
    reward = 0
    
    # Base step penalty, scaled to encourage efficient exploration
    # Higher penalty per step when there are fewer cells remaining to explore
    exploration_progress = total_covered_cells / coverable_cells
    if exploration_progress > 0.5:
        # Increase penalty as exploration progresses to discourage lingering
        reward -= 5  # Increased from 3 to make the agent move more urgently
    else:
        reward -= 2  # Increased from 1 to make the agent move more purposefully
    
    # Major reward for exploring new cells - scale higher as cells get harder to find
    if new_cell_covered:
        # Exponentially increase reward as exploration progresses
        # This encourages finding those last few difficult cells
        reward += 150 * (1 + 3 * exploration_progress**2)  # Increased multiplier from 2 to 3
    
    # Severe penalty for getting caught
    if game_over:
        reward -= 1000
        reset_timestep()
        danger_table_set = False
        return reward
    
    # Completion bonus with time efficiency factor
    if cells_remaining == 0:
        time_efficiency_bonus = steps_remaining / 500.0  # Higher bonus for faster completion
        reward += 3000 + (1500 * time_efficiency_bonus)  # Increased from 2000/1000 to 3000/1500
        return reward
    
    # Convert agent_pos to x,y coordinates
    agent_x = agent_pos % 10
    agent_y = agent_pos // 10
    
    # Predictive danger rewards - visit cells that will be dangerous soon
    cur_timestep = get_timestep()
    next_steps = [(cur_timestep + 1) % 4, (cur_timestep + 2) % 4, (cur_timestep + 3) % 4]
    
    # Add reward for being in cells that will become dangerous soon
    danger_reward = 0
    for index, step in enumerate(next_steps):
        if danger_table[agent_x][agent_y][step] == 1:
            # Higher reward for cells that will be dangerous sooner
            danger_reward += 15 * (3 - index)  # Increased from 10 to 15
    reward += danger_reward
    
    # Add an exploration incentive to encourage visiting new areas
    # This encourages the agent to move toward areas with many unexplored cells
    if cells_remaining > 0:
        # Higher reward when more cells still need to be explored
        exploration_urgency = cells_remaining / coverable_cells
        reward += 30 * exploration_urgency  # Increased from 20 to 30
    
    # Track positions and provide strong penalties for revisiting the same areas
    global last_positions
    if 'last_positions' not in globals():
        # Initialize if not already done
        last_positions = []
    
    # Add current position to history
    last_positions.append((agent_x, agent_y))
    # Keep only the last 30 positions (increased from 20)
    if len(last_positions) > 30:
        last_positions.pop(0)
    
    # Check for repeated positions with a stronger penalty
    position_counts = {}
    for pos in last_positions:
        if pos in position_counts:
            position_counts[pos] += 1
        else:
            position_counts[pos] = 1
    
    # Much stronger penalties for repeating positions
    max_count = max(position_counts.values()) if position_counts else 0
    if max_count > 3:  # Reduced threshold from 5 to 3
        reward -= (max_count - 3) * 15  # Increased penalty from 5 to 15
    
    # Special handling for known problem area (10,3)
    # The 10,3 position is out of bounds (grid is 0-9), but we'll check for position (9,3)
    if agent_x == 9 and agent_y == 3:
        # Apply a strong penalty when the agent is at this problem spot
        reward -= 50
        
        # Count how many times the agent has been at this position recently
        stuck_count = sum(1 for pos in last_positions[-10:] if pos == (9, 3))
        if stuck_count > 2:
            # Apply an escalating penalty if the agent keeps coming back here
            reward -= stuck_count * 25
    
    # Add a momentum factor to discourage back-and-forth movement
    if len(last_positions) >= 4:
        # Check if agent is just moving back and forth between two positions
        last_four = last_positions[-4:]
        if (last_four[0] == last_four[2]) and (last_four[1] == last_four[3]):
            # Penalize this oscillating behavior
            reward -= 40
    
    # Track unique positions visited in the last N steps to reward diverse exploration
    recent_positions = last_positions[-15:]
    unique_positions = len(set(recent_positions))
    # Reward for visiting more unique positions recently
    reward += unique_positions * 2
    
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

def set_observation_space(obs_space_number):
    """
    Set which observation space to use (1, 2, 3, or 4)
    """
    global DEFAULT_OBSERVATION_SPACE
    if obs_space_number in [1, 2, 3, 4]:
        DEFAULT_OBSERVATION_SPACE = obs_space_number
        print(f"Using observation space {obs_space_number}")
    else:
        print(f"Invalid observation space number: {obs_space_number}. Using default (1)")
        DEFAULT_OBSERVATION_SPACE = 1
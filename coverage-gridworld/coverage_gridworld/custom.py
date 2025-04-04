import numpy as np
import gymnasium as gym

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

# From env.py because I couldn't figure out how to actually import them.
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

# Store danger table here within custom.py
danger_table = [[[0 for _ in range(4)] for _ in range(10)] for _ in range(10)]
danger_table_set = False
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


def reward(info: dict) -> float:
    """
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode.
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step
    - game_over (bool) : if the game was terminated because the player was seen by an enemy or not
    """
    enemies = info["enemies"]
    agent_pos = info["agent_pos"]
    total_covered_cells = info["total_covered_cells"]
    cells_remaining = info["cells_remaining"]
    coverable_cells = info["coverable_cells"]
    steps_remaining = info["steps_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    global danger_table_set
    if not danger_table_set:
        set_danger_table(determine_cell_danger_tables(enemies))
    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using
    reward = 0
    reward += (100 - coverable_cells) if new_cell_covered else -coverable_cells
    reward -= 1000 if game_over else 0

    # Try modifying the reward to reward cells which are dangerous SOMETIMES, but not right now
    # Logic here is that you want to traverse the harder cells while you can
    # Reward cells that will be dangerous 3 steps from now the most
    cur_timestep = get_timestep()
    next_steps = [(cur_timestep + 1) % 4, (cur_timestep + 2) % 4, (cur_timestep + 3) % 4]
    for index, i in enumerate(next_steps):
        reward += danger_table[agent_pos % 10][agent_pos // 10][i] * 5 * (index + 1)
    incr_timestep()
    if game_over:
        reset_timestep()
        danger_table_set = False
    return reward

timestep = 0

def get_timestep():
    global timestep
    return timestep

def incr_timestep():
    global timestep
    timestep += 1
    print("Timestep incremented")

def reset_timestep():
    global timestep
    timestep = 0
    print("Timestep reset")

def set_danger_table(new_table):
    global danger_table
    danger_table = new_table
    global danger_table_set
    danger_table_set = True
    print(danger_table)
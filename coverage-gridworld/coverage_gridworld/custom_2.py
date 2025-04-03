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

def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
    """
    # The grid has (10, 10, 3) shape and can store values from 0 to 255 (uint8). To use the whole grid as the
    # observation space, we can consider a MultiDiscrete space with values in the range [0, 256).

    # Make a grid with the symbolic representation, rather than color.
    # In this case, we simplify such that the first layer encodes the agent, a covered cell, an uncovered cell, or an untraversable tile.
    cell_values = np.zeros(shape=(10, 10), dtype=np.uint8) + 4
    danger_values = np.zeros(shape=(10, 10), dtype=np.uint8) + 5
    cell_values = np.stack((cell_values, danger_values), axis=-1)
    # if MultiDiscrete is used, it's important to flatten() numpy arrays!
    return gym.spaces.MultiDiscrete(cell_values.flatten())


def observation(grid: np.ndarray):
    """
    Function that returns the observation for the current state of the environment.
    """
    # If the observation returned is not the same shape as the observation_space, an error will occur!
    # Make sure to make changes to both functions accordingly.

    cell_values = np.zeros(shape=(10, 10), dtype=np.uint8)
    danger_values = np.zeros(shape=(10, 10), dtype=np.uint8)

    enemies = []
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            symbol = tuple(grid[x][y])
            symbol_num = list(COLOR_IDS.keys())[list(COLOR_IDS.values()).index(symbol)]
            match symbol_num:
                case 0, 5:
                    cell_values[x][y] = 0
                case 1, 6:
                    cell_values[x][y] = 1
                case 2, 4:
                    cell_values[x][y] = 2
                case 3:
                    cell_values[x][y] = 3
            # Keep track of any cells containing enemies, for later.
            if tuple(grid[x][y]) == GREEN:
                enemies.append((x, y))
            # Doing this means we can reduce the danger factor by one;
            # at least one enemy won't be viewing this cell next step.
            if tuple(grid[x][y]) == RED or tuple(grid[x][y]) == LIGHT_RED:
                danger_values[x][y] = 4

    for (ex, ey) in enemies:
        for (x_, y_)in [(0, -1), (-1, 0), (0, 1), (1, 0)]:

            for r in range(1, 5):
                cx, cy = (ex + r*x_, ey + r*y_)
                # Make sure the cell is in bounds
                if cx not in range(10) or cy not in range(10):
                    break
                # Then, check if vision is obstructed
                if tuple(grid[cx][cy]) == BROWN or tuple(grid[cx][cy]) == GREEN:
                    break
                # Otherwise, set a danger value for the current cell based on if it's being actively observed.
                # By taking it mod 5, we can account for the value from earlier.
                danger_values[cx][cy] = (danger_values[cx][cy] + 1) % 5

    cell_values = np.stack((cell_values, danger_values), axis=-1)
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

    # IMPORTANT: You may design a reward function that uses just some of these values. Experiment with different
    # rewards and find out what works best for the algorithm you chose given the observation space you are using
    reward = 0
    reward += 10 if new_cell_covered else -5
    reward -= 99999 if game_over else 0
    return reward

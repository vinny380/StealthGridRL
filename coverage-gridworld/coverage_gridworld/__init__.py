from gymnasium.envs.registration import register
from coverage_gridworld.env import CoverageGridworld

register(
    id="standard",
    entry_point="coverage_gridworld:CoverageGridworld",
    max_episode_steps=250
)

register(
    id="maze",   # easy difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    max_episode_steps=250,
    kwargs={
        "predefined_map": [
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
        ]
    }
)

register(
    id="chokepoint",   # medium difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    max_episode_steps=250,
    kwargs={
        "predefined_map": [
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
        ]
    }
)

register(
    id="sneaky_enemies",   # hard difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    max_episode_steps=250,
    kwargs={
        "predefined_map": [
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
    }
)

# To create a predefined map, just add walls and enemies. The agent always starts in the top-left corner.
# The enemy's orientation is randomly defined and the cells under surveillance will be spawned automatically

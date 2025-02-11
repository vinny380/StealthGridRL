# Coverage Gridworld

![visualization](media/sneaky_enemies.gif "Sneaky Enemies sample layout")

## Overview

The goal of the Agent (Grey circle) is to explore all available cells within the map as quickly as possible without 
being seen by enemies. 

Black cells have not yet been explored and White cells already have. While moving, the Agent should not move towards 
a Wall (Brown) or an Enemy (Green), otherwise it will get stunned (Yellow) and be unable to move for 1 round (step).

Also, the Enemies are on the lookout for the agent, constantly surveilling their surrounding area (Red/Light Red). 
All Enemies have a fixed range that they can observe, and they keep rotating counter-clockwise at every step. If the
Agent is seen by an Enemy, the mission fails.

## Installation

To install the environment, simply run 

```bash
pip install -e coverage-gridworld
```

## Map modes

There are three ways of defining the map layouts to be used:

### Standard maps

Three standard maps are included in the `\coverage_gridworld\__init.py__` file: 
- `maze`: easy difficulty map, 2 enemies and focuses mostly on movement,
- `chokepoint`: medium difficulty map, 4 enemies and requires precise movement and timing,
- `sneaky_enemies`: hard difficulty map, 5 enemies and many walls, with some cells being surveilled by multiple 
enemies.

The standard maps can be used by using their tag on `gymnasium.make()`. For example:

```python
gymnasium.make("sneaky_enemies", render_mode="human", predefined_map_list=None)
```

If a standard map is selected, then it will be used for every episode of training.

### Random maps

If the `standard` tag is used in `gymnasium.make()`, then random maps will be generated at every new episode.

Random map creation follows certain rules, such as having every `BLACK` cell reachable by the agent, but due to 
randomness, some of the maps created may be impossible to be fully explored (e.g. a cell is under constant surveillance
by 4 different enemies).

```python
gymnasium.make("standard", render_mode="human", predefined_map_list=None)
```

### Predefined map list

If you wish to have finer control of the training process of the agent, a list of predefined maps can be created and
used with `gymnasium.make()`:

```python
gymnasium.make("standard", render_mode="human", predefined_map_list=maps)
```

An example of such a list is provided in the `main.py` file.

To create a list of maps, just copy one of the provided examples and modify the values according to their color IDs:
- `3` - `GREY` -> agent (must always be positioned at cell `(0, 0)`),
- `2` - `BROWN` -> wall (walls cannot enclose an area, causing a cell to be out of reach of the agent),
- `4` - `GREEN` -> enemy (the enemy FOV cells are placed automatically by the environment and their starting orientation
is randomly determined),
- `0` - `BLACK` -> cells to be explored.

Any other color ID used will be ignored by the environment and a value of `0` will be assigned in its place.

## MDP

### Action Space

The action is discrete in the range `{0, 4}`.

- 0: Move left
- 1: Move down
- 2: Move right
- 3: Move up
- 4: Stay (do not move)

### Observation Space

The observation is an `N * N * 3` `uint8` array, where `N` is the grid dimension (standard value `10`) and `3` 
corresponds to the RGB channels. Each RGB channel has a range of `0` to `255` (inclusive), making the observation space
MultiDiscrete.

### Starting State
The episode starts with the agent at the top-left tile `(0, 0)`, with that tile already explored.

### Transition
The transitions are deterministic. If the agent moves towards a wall or enemy cell, it will be stunned for the next
step, and whatever action while stunned will have no effect.

### Rewards
The standard reward returned by the environment is the score from the game. Score is calculated by the following 
equation: 

``SCORE = num_tiles_covered * 5 + is_alive * time_remaining`` 

(`is_alive` is a boolean that assumes value `0` if the player died at the end of the episode or `1` if it was alive)
        
However, novel reward schemes may be implemented on the agent side, penalizing or rewarding certain behaviors (e.g. 
hitting a wall, not moving, walking over an explored cell, etc.). The `info` dictionary returned by the step method may 
be used for that and contains the following keys:

- `enemies` : `list` of `Enemy` objects,
- `agent_pos` : agent position as an `int`, considering the flattened grid (e.g. cell `(2, 3)` corresponds to 
position `23`),
- `covered_cells` : `int`, how many cells have been covered by the agent so far,
- `coverable_cells` : `int`, how many cells can be covered in the current map layout,
- `steps_remaining` : steps remaining in the episode,
- `is_stuned` : `boolean` value determining whether the agent is stunned or not.

### Episode End

By default, an episode ends if any of the following happens:
- The player dies (gets spotted by an enemy),
- Explores all tiles,
- Time runs out.


## Testing

Two functions are provided within `main.py` for quick testing of the environment: 

* `human_player()`, where the agent moves according to user inputs (WASD for directions and E for `STAY`),
* `random_player()`, for quick visualization of a randomized policy.

Both functions return the `action` variable that can be used with the `step()` function of the environment.
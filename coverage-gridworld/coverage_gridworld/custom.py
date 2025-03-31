import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from typing import Dict, List, Optional, Tuple
import os
import json
import time
from datetime import datetime
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

# Global color constants (matching those from env.py)
BLACK = (0, 0, 0)            # unexplored cell
WHITE = (255, 255, 255)      # explored cell
BROWN = (101, 67, 33)        # wall
GREY = (160, 161, 161)       # agent
GREEN = (31, 198, 0)         # enemy
RED = (255, 0, 0)            # unexplored cell being observed by an enemy
LIGHT_RED = (255, 127, 127)  # explored cell being observed by an enemy

# Define action constants
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4


class GridObservationWrapper(gym.ObservationWrapper):
    """
    Observation Space 1: Grid-Based Direct Representation
    Represents the environment as a multi-channel grid with separate channels for:
    - Agent position (one-hot)
    - Walls (binary)
    - Explored cells (binary)
    - Enemy positions (binary)
    - Enemy FOV (binary)
    - Distance to nearest unexplored cell (normalized)
    """
    def __init__(self, env):
        super().__init__(env)
        # Define the new observation space: 6 channels, 10x10 grid
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(6, 10, 10), 
            dtype=np.float32
        )
        self.grid_size = 10
        
    def observation(self, grid):
        """
        Converts the raw grid into a multi-channel representation
        """
        grid_size = self.grid_size
        agent_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
        wall_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
        explored_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
        enemy_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
        enemy_fov_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
        distance_channel = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Reshape the grid based on its shape
        if len(grid.shape) == 1:  # Flattened grid
            if grid.shape[0] == grid_size * grid_size * 3:
                # Reshape to 10x10x3
                reshaped_grid = grid.reshape(grid_size, grid_size, 3)
            else:
                # If shape is unexpected, try to reshape it intelligently or use zeros
                print(f"Warning: Unexpected grid shape: {grid.shape}")
                reshaped_grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        else:
            # If already properly shaped, no need to reshape
            reshaped_grid = grid
        
        # Initialize agent position
        agent_pos = None
        
        # Find unexplored cells
        unexplored_cells = []
        
        # Fill channels based on grid colors
        for i in range(grid_size):
            for j in range(grid_size):
                cell_color = reshaped_grid[i, j]
                
                # Agent position (GREY)
                if np.array_equal(cell_color, GREY):
                    agent_channel[i, j] = 1
                    explored_channel[i, j] = 1
                    agent_pos = (i, j)
                
                # Wall (BROWN)
                elif np.array_equal(cell_color, BROWN):
                    wall_channel[i, j] = 1
                
                # Enemy (GREEN)
                elif np.array_equal(cell_color, GREEN):
                    enemy_channel[i, j] = 1
                
                # Explored cells (WHITE)
                elif np.array_equal(cell_color, WHITE):
                    explored_channel[i, j] = 1
                
                # Enemy FOV cells (RED or LIGHT_RED)
                elif np.array_equal(cell_color, RED) or np.array_equal(cell_color, LIGHT_RED):
                    enemy_fov_channel[i, j] = 1
                    
                    # RED is unexplored under enemy surveillance
                    if np.array_equal(cell_color, RED):
                        unexplored_cells.append((i, j))
                
                # Unexplored cells (BLACK)
                elif np.array_equal(cell_color, BLACK):
                    unexplored_cells.append((i, j))
        
        # Calculate distance to nearest unexplored cell
        if agent_pos and unexplored_cells:
            for i in range(grid_size):
                for j in range(grid_size):
                    min_distance = float('inf')
                    for cell in unexplored_cells:
                        dist = math.sqrt((i - cell[0])**2 + (j - cell[1])**2)
                        min_distance = min(min_distance, dist)
                    # Normalize distance to [0, 1]
                    distance_channel[i, j] = min(1.0, min_distance / (2 * grid_size))
        
        # Stack all channels
        observation = np.stack([
            agent_channel, 
            wall_channel, 
            explored_channel, 
            enemy_channel, 
            enemy_fov_channel, 
            distance_channel
        ])
        
        return observation


class VectorObservationWrapper(gym.ObservationWrapper):
    """
    Observation Space 2: Feature-Based Vector Representation
    Creates a fixed-length vector containing:
    - Relative positions of nearby walls
    - Relative positions and orientations of enemies
    - Predicted next positions of enemy FOV
    - Relative positions of unexplored cells
    - Percentage of map explored
    - Danger level indicators (proximity to enemy FOV)
    - Safety corridor indicators
    - Distance-to-completion estimate
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Define the new observation space
        # - 8 wall proximity sensors (up, down, left, right, diagonals)
        # - For up to 5 enemies: position (x, y), orientation, and rotation direction (4) = 20 values
        # - 8 indicators for predicted enemy FOV movement
        # - 8 unexplored cell proximity indicators
        # - Percentage of map explored
        # - 5 danger level indicators (proximity to enemy FOV)
        # - 4 safety corridor indicators (safe directions)
        # - Distance-to-completion estimate
        vector_size = 8 + 20 + 8 + 8 + 1 + 5 + 4 + 1
        
        self.observation_space = spaces.Box(
            low=-1, high=1,
            shape=(vector_size,),
            dtype=np.float32
        )
        
        self.max_enemies = 5
        self.grid_size = 10
    
    def get_agent_position(self, grid_3d):
        """Extract agent position from grid"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if np.array_equal(grid_3d[i, j], GREY):
                    return (i, j)
        return None  # Should never happen
    
    def predict_enemy_fov_movement(self, enemies):
        """Predict the next position of enemy FOV cells based on their rotation pattern"""
        predicted_fov = []
        
        # Default FOV distance if not available in env
        enemy_fov_distance = getattr(self.env.unwrapped, 'enemy_fov_distance', 3)
        
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
                if (0 <= fov_y < self.grid_size and 0 <= fov_x < self.grid_size and
                    not np.array_equal(self.env.unwrapped.grid[fov_y, fov_x], BROWN) and
                    not np.array_equal(self.env.unwrapped.grid[fov_y, fov_x], GREEN)):
                    predicted_fov.append((fov_y, fov_x))
                else:
                    break  # Stop if obstacle or boundary reached
                    
        return predicted_fov
    
    def detect_safety_corridors(self, agent_pos, fov_cells, grid_3d):
        """
        Detect directions that are safe to move in (away from enemy FOV)
        Returns a list of 4 values indicating safety level in cardinal directions
        """
        if agent_pos is None:
            return [0, 0, 0, 0]
            
        agent_y, agent_x = agent_pos
        safe_directions = [1.0, 1.0, 1.0, 1.0]  # UP, RIGHT, DOWN, LEFT - higher is safer
        
        # Check each cardinal direction
        cardinal_dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT
        
        for dir_idx, (dy, dx) in enumerate(cardinal_dirs):
            # Check 3 steps in this direction
            for steps in range(1, 4):
                ny, nx = agent_y + dy * steps, agent_x + dx * steps
                
                # Check if out of bounds or wall
                if not (0 <= ny < self.grid_size and 0 <= nx < self.grid_size):
                    safe_directions[dir_idx] = 0.0  # Not safe - it's a boundary
                    break
                    
                if np.array_equal(grid_3d[ny, nx], BROWN):
                    safe_directions[dir_idx] = 0.0  # Not safe - it's a wall
                    break
                
                # Check if this position is close to FOV
                for fov_y, fov_x in fov_cells:
                    distance = math.sqrt((ny - fov_y)**2 + (nx - fov_x)**2)
                    if distance < 2.0:  # Closer than 2 cells is dangerous
                        # Reduce safety level based on proximity
                        safe_directions[dir_idx] = min(safe_directions[dir_idx], distance / 2.0)
        
        return safe_directions
    
    def estimate_completion_distance(self, agent_pos, unexplored_cells, total_covered, coverable_cells):
        """
        Estimate how far the agent is from completing the task
        Returns a normalized value where 0 means far from completion, 1 means close to completion
        """
        if not unexplored_cells or agent_pos is None:
            # If no unexplored cells or agent not found, return based on coverage
            return total_covered / coverable_cells
            
        agent_y, agent_x = agent_pos
        
        # Calculate minimum spanning tree distance (approximation)
        # Start with distance to closest unexplored cell
        min_distance = float('inf')
        for y, x in unexplored_cells:
            dist = math.sqrt((agent_y - y)**2 + (agent_x - x)**2)
            min_distance = min(min_distance, dist)
        
        # Normalize: 0 means far from completion, 1 means close to completion
        normalized_distance = 1.0 - min(1.0, min_distance / (2 * self.grid_size))
        
        # Combine with coverage percentage
        coverage_percentage = total_covered / coverable_cells
        
        # Weight more towards coverage as we explore more
        return 0.7 * coverage_percentage + 0.3 * normalized_distance
    
    def observation(self, grid):
        """
        Converts the raw grid into a feature vector
        """
        # Reshape the grid based on its shape
        if len(grid.shape) == 1:  # Flattened grid
            if grid.shape[0] == self.grid_size * self.grid_size * 3:
                # Reshape to 10x10x3
                reshaped_grid = grid.reshape(self.grid_size, self.grid_size, 3)
            else:
                # If shape is unexpected, use zeros
                print(f"Warning: Unexpected grid shape: {grid.shape}")
                reshaped_grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        else:
            # If already properly shaped, no need to reshape
            reshaped_grid = grid
        
        agent_pos = self.get_agent_position(reshaped_grid)
        
        if not agent_pos:
            # If agent not found, return zeros
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        agent_y, agent_x = agent_pos
        
        # 1. Wall proximity sensors (8 directions)
        wall_sensors = np.zeros(8, dtype=np.float32)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for i, (dy, dx) in enumerate(directions):
            # Set initial distance to maximum (normalized to 1)
            wall_sensors[i] = 1.0
            
            # Check for walls in this direction (up to 3 cells away)
            for distance in range(1, 4):
                y, x = agent_y + dy * distance, agent_x + dx * distance
                
                # Check if out of bounds
                if not (0 <= y < self.grid_size and 0 <= x < self.grid_size):
                    wall_sensors[i] = distance / 3.0  # Normalize to [0, 1]
                    break
                
                # Check if wall
                if np.array_equal(reshaped_grid[y, x], BROWN):
                    wall_sensors[i] = distance / 3.0  # Normalize to [0, 1]
                    break
        
        # 2. Enemy positions and orientations
        enemy_features = np.zeros(4 * self.max_enemies, dtype=np.float32)
        enemies = []
        
        # Find enemies
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if np.array_equal(reshaped_grid[i, j], GREEN):
                    enemies.append((i, j))
                    if len(enemies) >= self.max_enemies:
                        break
        
        # Get enemy orientations from env
        enemies_with_orientation = []
        for e_idx, enemy_obj in enumerate(self.env.enemy_list):
            if e_idx < len(enemies):
                # Also determine rotation direction (0 for clockwise, 1 for counter-clockwise)
                # Default to counter-clockwise (1)
                rotation_direction = 1  
                enemies_with_orientation.append((enemy_obj.y, enemy_obj.x, enemy_obj.orientation, rotation_direction))
        
        # Fill enemy features
        for i, (y, x, orientation, rotation) in enumerate(enemies_with_orientation):
            if i >= self.max_enemies:
                break
                
            # Relative normalized position
            enemy_features[i*4] = (y - agent_y) / self.grid_size  # y distance
            enemy_features[i*4+1] = (x - agent_x) / self.grid_size  # x distance
            
            # Orientation (normalized)
            enemy_features[i*4+2] = orientation / 3.0  # Normalize to [0, 1]
            
            # Rotation direction
            enemy_features[i*4+3] = rotation
        
        # 3. Predicted enemy FOV movement
        enemy_fov_predictions = np.zeros(8, dtype=np.float32)
        predicted_fov_by_direction = self.predict_enemy_fov_movement(self.env.enemy_list)
        
        # For each direction, use the count of predicted FOV cells as an indicator
        for i, fov_cells in enumerate(predicted_fov_by_direction):
            # Normalize by setting a reasonable maximum (e.g., 5 FOV cells)
            enemy_fov_predictions[i] = min(1.0, len(fov_cells) / 5.0)
        
        # 4. Unexplored cell indicators
        unexplored_indicators = np.zeros(8, dtype=np.float32)
        unexplored_cells = []
        
        # Find unexplored cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (np.array_equal(reshaped_grid[i, j], BLACK) or 
                    np.array_equal(reshaped_grid[i, j], RED)):
                    unexplored_cells.append((i, j))
        
        # For each direction, find closest unexplored cell
        for i, (dy, dx) in enumerate(directions):
            closest_distance = float('inf')
            
            for y, x in unexplored_cells:
                # Check if this cell is roughly in the direction we're looking
                cell_dy, cell_dx = y - agent_y, x - agent_x
                
                # Dot product to check if vectors are pointing in similar direction
                dot_product = dy * cell_dy + dx * cell_dx
                
                if dot_product > 0:  # Cell is in this general direction
                    distance = math.sqrt((y - agent_y)**2 + (x - agent_x)**2)
                    closest_distance = min(closest_distance, distance)
            
            # Normalize and invert (closer = higher value)
            if closest_distance != float('inf'):
                # Normalize to [0, 1] where 1 means very close
                unexplored_indicators[i] = 1.0 - min(1.0, closest_distance / self.grid_size)
            else:
                unexplored_indicators[i] = 0.0
        
        # 5. Percentage of map explored
        total_cells = self.env.coverable_cells
        explored_cells = self.env.total_covered_cells
        exploration_percentage = np.array([explored_cells / total_cells], dtype=np.float32)
        
        # 6. Danger indicators (proximity to enemy FOV)
        danger_indicators = np.zeros(5, dtype=np.float32)
        
        # Find FOV cells
        fov_cells = []
        for enemy in self.env.enemy_list:
            fov_cells.extend(enemy.get_fov_cells())
        
        # Calculate danger level based on distance to closest FOV cell
        if fov_cells:
            distances = []
            for fov_y, fov_x in fov_cells:
                distance = math.sqrt((fov_y - agent_y)**2 + (fov_x - agent_x)**2)
                distances.append(distance)
            
            # Sort distances
            distances.sort()
            
            # Fill danger indicators with normalized distances to the closest FOV cells
            for i in range(min(5, len(distances))):
                # Normalize and invert (closer = higher danger)
                danger_indicators[i] = 1.0 - min(1.0, distances[i] / self.grid_size)
        
        # 7. Safety corridor indicators
        safety_corridors = self.detect_safety_corridors(agent_pos, fov_cells, reshaped_grid)
        
        # 8. Distance to completion estimate
        completion_distance = np.array([self.estimate_completion_distance(
            agent_pos, unexplored_cells, explored_cells, total_cells)], dtype=np.float32)
        
        # Combine all features
        observation = np.concatenate([
            wall_sensors,
            enemy_features,
            enemy_fov_predictions,
            unexplored_indicators,
            exploration_percentage,
            danger_indicators,
            safety_corridors,
            completion_distance
        ])
        
        return observation


# Default observation space for backward compatibility
def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Default observation space (flattened grid)
    """
    # Create a space for RGB values (0-255) for each cell in the grid
    return gym.spaces.Box(
        low=0, 
        high=255,
        shape=(env.grid_size * env.grid_size * 3,),  # Flattened shape
        dtype=np.uint8
    )


# Default observation function for backward compatibility
def observation(grid: np.ndarray):
    """
    Default observation function (returns flattened grid)
    """
    return grid


class ExplorationRewardWrapper(gym.Wrapper):
    """
    Reward Function 1: Exploration-Focused
    - Positive reward for each new cell explored
    - Small negative step penalty to encourage efficiency
    - Large penalty for being detected
    - Completion bonus proportional to explored area percentage
    """
    def __init__(self, env, exploration_reward=1.0, step_penalty=0.01, 
                 detection_penalty=10.0, completion_bonus_factor=10.0):
        super().__init__(env)
        self.exploration_reward = exploration_reward
        self.step_penalty = step_penalty
        self.detection_penalty = detection_penalty
        self.completion_bonus_factor = completion_bonus_factor
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract info
        new_cell_covered = info["new_cell_covered"]
        game_over = info["game_over"]
        total_covered_cells = info["total_covered_cells"]
        coverable_cells = info["coverable_cells"]
        
        # Calculate custom reward
        custom_reward = 0.0
        
        # Reward for exploring new cells
        if new_cell_covered:
            custom_reward += self.exploration_reward
        
        # Penalty per step to encourage efficiency
        custom_reward -= self.step_penalty
        
        # Large penalty if detected by enemy
        if game_over and total_covered_cells < coverable_cells:
            custom_reward -= self.detection_penalty
        
        # Bonus for completion based on percentage explored
        if terminated and not game_over:
            exploration_percentage = total_covered_cells / coverable_cells
            custom_reward += self.completion_bonus_factor * exploration_percentage
            
        return obs, custom_reward, terminated, truncated, info


class SafetyRewardWrapper(gym.Wrapper):
    """
    Reward Function: Balanced Exploration Focus
    - Strong positive reward for each new cell explored
    - Moderate penalty for being detected
    - Small penalty for being near enemy FOV
    - Large bonus for complete exploration
    - Progressive bonus for increasing coverage
    - Small penalty for revisiting cells
    - Strong penalty for staying in place
    """
    def __init__(self, env):
        super().__init__(env)
        self.detection_penalty = 100.0  # Significant penalty for getting caught
        self.proximity_penalty_factor = 0.1  # Small penalty for being near enemies
        self.exploration_reward = 20.0  # Increased reward for finding new cells
        self.revisit_penalty = 0.5  # Increased penalty for revisiting cells
        self.completion_bonus = 500.0  # Large bonus for completing the map
        self.coverage_bonus = 50.0  # Bonus for reaching coverage milestones
        self.grid_size = 10
        self.unexplored_cells = []
        self.last_min_distance = None
        self.visited_cells = set([(0, 0)])  # Start with agent's initial position as visited
        self.last_coverage = 0  # Track last coverage milestone reached
        self.position_history = []  # Track recent positions
        self.stagnation_counter = 0  # Counter for steps without movement
        self.last_position = None  # Track last position
        self.stay_penalty = 1.0  # Penalty for staying in place
        self.stagnation_penalty = 0.5  # Penalty for getting stuck
        
        # Update observation space to match model's expectations
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(602,),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        """Reset wrapper state when environment resets"""
        obs, info = self.env.reset(**kwargs)
        self.unexplored_cells = []
        self.last_min_distance = None
        self.visited_cells = set([(0, 0)])  # Reset visited cells
        self.last_coverage = 0  # Reset coverage milestone
        self.position_history = []  # Reset position history
        self.stagnation_counter = 0  # Reset stagnation counter
        self.last_position = None  # Reset last position
        # Convert observation to match expected format
        processed_obs = self._process_observation(obs)
        return processed_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            return self._process_observation(obs), reward, terminated, truncated, info

        # Extract information
        enemies = info["enemies"]
        agent_pos = info["agent_pos"]
        total_covered_cells = info["total_covered_cells"]
        coverable_cells = info["coverable_cells"]
        new_cell_covered = info["new_cell_covered"]
        
        # Calculate agent's coordinates
        agent_y, agent_x = agent_pos // self.grid_size, agent_pos % self.grid_size
        current_position = (agent_y, agent_x)
        
        # Update position tracking
        self.position_history.append(current_position)
        if len(self.position_history) > 10:  # Keep last 10 positions
            self.position_history.pop(0)
        
        # Check for stagnation
        if self.last_position == current_position:
            self.stagnation_counter += 1
            custom_reward = -self.stay_penalty  # Immediate penalty for staying in place
        else:
            self.stagnation_counter = 0
            custom_reward = 0.0
        
        # Update last position
        self.last_position = current_position
        
        # Update visited cells set
        self.visited_cells.add(current_position)
        
        # Strong reward for exploring new cells
        if new_cell_covered:
            custom_reward += self.exploration_reward
            self.stagnation_counter = 0  # Reset stagnation counter
            
            # Check for coverage milestones (25%, 50%, 75%, 100%)
            current_coverage = total_covered_cells / coverable_cells
            coverage_milestones = [0.25, 0.5, 0.75, 1.0]
            
            for milestone in coverage_milestones:
                if current_coverage >= milestone and self.last_coverage < milestone:
                    custom_reward += self.coverage_bonus
                    self.last_coverage = milestone
                    break
        
        # Calculate distances to nearest unexplored cells
        self.update_unexplored_cells(self.env.unwrapped.grid)
        distance_to_nearest = self.get_min_distance_to_unexplored(agent_y, agent_x)
        
        # Reward for moving toward unexplored cells
        if self.last_min_distance is not None and distance_to_nearest < self.last_min_distance:
            custom_reward += 0.5  # Small reward for getting closer to unexplored areas
        
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
        
        # Small penalty for proximity to danger
        if min_distance < 3:
            danger_factor = np.exp(3 - min_distance) - 1
            custom_reward -= self.proximity_penalty_factor * danger_factor

        # Small penalty for revisiting cells
        if not new_cell_covered and current_position in self.visited_cells:
            custom_reward -= self.revisit_penalty
            
        # Additional stagnation penalty if stuck for too long
        if self.stagnation_counter > 5:
            custom_reward -= self.stagnation_penalty * self.stagnation_counter

        # Bonus for completion based on percentage explored
        if terminated and not info.get("game_over", False):
            exploration_percentage = total_covered_cells / coverable_cells
            custom_reward += self.completion_bonus * exploration_percentage
            
        return self._process_observation(obs), custom_reward, terminated, truncated, info

    def _process_observation(self, obs):
        """Convert observation to the expected format"""
        # First flatten the observation
        if isinstance(obs, np.ndarray):
            flattened = obs.reshape(-1)
        else:
            flattened = np.array(obs).reshape(-1)
        
        # Convert to float32 and normalize to [0, 1]
        normalized = flattened.astype(np.float32) / 255.0
        
        # Add extra features
        extra_features = np.zeros(302, dtype=np.float32)
        
        # Combine normalized observation with extra features
        return np.concatenate([normalized, extra_features])

    def get_min_distance_to_wall(self, agent_y, agent_x):
        """Calculate the minimum distance to any wall"""
        min_distance = float('inf')
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = agent_y + dy, agent_x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    if np.array_equal(self.env.unwrapped.grid[ny, nx], (101, 67, 33)):  # BROWN color for walls
                        distance = (dy**2 + dx**2)**0.5
                        min_distance = min(min_distance, distance)
        return min_distance if min_distance != float('inf') else 0

    def update_unexplored_cells(self, grid_3d):
        """Update the list of unexplored cells"""
        self.unexplored_cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (np.array_equal(grid_3d[i, j], BLACK) or 
                    np.array_equal(grid_3d[i, j], RED)):
                    self.unexplored_cells.append((i, j))
    
    def get_min_distance_to_unexplored(self, agent_y, agent_x):
        """Calculate the minimum distance to any unexplored cell"""
        if not self.unexplored_cells:
            return 0
            
        min_distance = float('inf')
        for cell_y, cell_x in self.unexplored_cells:
            distance = math.sqrt((cell_y - agent_y)**2 + (cell_x - agent_x)**2)
            min_distance = min(min_distance, distance)
        
        return min_distance


class EfficiencyRewardWrapper(gym.Wrapper):
    """
    Reward Function 3: Efficiency-Focused
    - Positive reward for each new cell explored
    - Dynamic step penalty that increases with time
    - Reward for moving toward unexplored areas
    - Penalty for revisiting already explored cells
    - Bonus for completion speed
    """
    def __init__(self, env, exploration_reward=1.0, step_penalty_factor=0.01,
                 direction_reward=0.5, revisit_penalty=0.2, speed_bonus_factor=50.0):
        super().__init__(env)
        self.exploration_reward = exploration_reward
        self.step_penalty_factor = step_penalty_factor
        self.direction_reward = direction_reward
        self.revisit_penalty = revisit_penalty
        self.speed_bonus_factor = speed_bonus_factor
        self.grid_size = 10
        self.step_count = 0
        self.last_agent_pos = None
        self.unexplored_cells = []
        self.last_distance_to_unexplored = float('inf')
        
    def reset(self, **kwargs):
        self.step_count = 0
        self.last_agent_pos = None
        self.unexplored_cells = []
        self.last_distance_to_unexplored = float('inf')
        return self.env.reset(**kwargs)
    
    def update_unexplored_cells(self, grid_3d):
        """Update the list of unexplored cells"""
        self.unexplored_cells = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (np.array_equal(grid_3d[i, j], BLACK) or 
                    np.array_equal(grid_3d[i, j], RED)):
                    self.unexplored_cells.append((i, j))
    
    def get_min_distance_to_unexplored(self, agent_y, agent_x):
        """Calculate the minimum distance to any unexplored cell"""
        if not self.unexplored_cells:
            return 0
            
        min_distance = float('inf')
        for cell_y, cell_x in self.unexplored_cells:
            distance = math.sqrt((cell_y - agent_y)**2 + (cell_x - agent_x)**2)
            min_distance = min(min_distance, distance)
        
        return min_distance
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract info
        new_cell_covered = info["new_cell_covered"]
        game_over = info["game_over"]
        total_covered_cells = info["total_covered_cells"]
        coverable_cells = info["coverable_cells"]
        agent_pos = info["agent_pos"]
        steps_remaining = info["steps_remaining"]
        
        # Update step count
        self.step_count += 1
        
        # Get agent's coordinates
        agent_y, agent_x = agent_pos // self.grid_size, agent_pos % self.grid_size
        
        # Reshape the observation to get the grid
        grid_3d = obs.reshape(self.grid_size, self.grid_size, 3)
        
        # Update unexplored cells list
        self.update_unexplored_cells(grid_3d)
        
        # Calculate distance to nearest unexplored cell
        current_distance = self.get_min_distance_to_unexplored(agent_y, agent_x)
        
        # Calculate custom reward
        custom_reward = 0.0
        
        # Reward for exploring new cells
        if new_cell_covered:
            custom_reward += self.exploration_reward
        
        # Dynamic step penalty (increasing with time)
        custom_reward -= self.step_penalty_factor * self.step_count
        
        # Reward for moving toward unexplored areas
        if self.last_agent_pos is not None:
            # Only if we have a previous position to compare with
            distance_reduction = self.last_distance_to_unexplored - current_distance
            if distance_reduction > 0:
                custom_reward += self.direction_reward * distance_reduction
        
        # Penalty for revisiting already explored cells
        if self.last_agent_pos is not None:
            last_y, last_x = self.last_agent_pos
            if not new_cell_covered and (agent_y != last_y or agent_x != last_x):
                # If the agent moved but didn't explore a new cell
                custom_reward -= self.revisit_penalty
        
        # Severe penalty if detected by enemy
        if game_over and total_covered_cells < coverable_cells:
            custom_reward -= 10.0  # Fixed penalty for detection
        
        # Bonus for completion speed
        if terminated and total_covered_cells == coverable_cells:
            # The faster the completion, the higher the bonus
            time_bonus = self.speed_bonus_factor * (1.0 / max(1, self.step_count))
            custom_reward += time_bonus
        
        # Update last position and distance
        self.last_agent_pos = (agent_y, agent_x)
        self.last_distance_to_unexplored = current_distance
            
        return obs, custom_reward, terminated, truncated, info


# Default reward function for backward compatibility
def reward(info: dict) -> float:
    """
    Default reward function (always returns 0)
    """
    return 0


class ExperimentCallback(BaseCallback):
    """
    Custom callback for experiment tracking
    """
    def __init__(self, eval_env, experiment_name, eval_freq=10000, 
                 n_eval_episodes=5, log_path="./logs", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.experiment_name = experiment_name
        self.log_path = log_path
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_path):
            os.makedirs(log_path)
            
        # Results tracking
        self.all_rewards = []
        self.eval_rewards = []
        self.success_rates = []
        self.exploration_rates = []
        self.episode_lengths = []
        self.timesteps = []
        
    def _on_step(self) -> bool:
        # Collect episode stats from the training
        if self.locals.get("dones") is not None:
            for info in self.locals.get("infos"):
                if "episode" in info:
                    # Extract info from Monitor wrapper
                    ep_reward = info["episode"]["r"]
                    ep_length = info["episode"]["l"]
                    self.all_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
        # Periodically evaluate the agent
        if self.num_timesteps % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            
            # Track timesteps
            self.timesteps.append(self.num_timesteps)
            
            # Track evaluation rewards
            self.eval_rewards.append(mean_reward)
            
            # Calculate success rate and exploration rate
            success_count = 0
            exploration_percentage = 0
            
            # Run additional episodes to collect more detailed stats
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                
                # Episode completed - extract metrics
                total_covered = info["total_covered_cells"]
                coverable_cells = info["coverable_cells"]
                game_over = info["game_over"]
                
                # Success = explored all cells without being detected
                if total_covered == coverable_cells and not game_over:
                    success_count += 1
                
                # Exploration percentage
                exploration_percentage += (total_covered / coverable_cells)
            
            # Calculate average exploration percentage and success rate
            exploration_percentage /= self.n_eval_episodes
            success_rate = success_count / self.n_eval_episodes
            
            self.success_rates.append(success_rate)
            self.exploration_rates.append(exploration_percentage)
            
            # Log results
            if self.verbose > 0:
                print(f"Timestep {self.num_timesteps}: Mean reward = {mean_reward:.2f}, "
                      f"Success rate = {success_rate:.2f}, "
                      f"Exploration = {exploration_percentage:.2f}")
            
            # Save results periodically
            self.save_results()
        
        return True
    
    def save_results(self):
        """Save experiment results to disk"""
        results = {
            "experiment_name": self.experiment_name,
            "timesteps": self.timesteps,
            "eval_rewards": self.eval_rewards,
            "success_rates": self.success_rates,
            "exploration_rates": self.exploration_rates,
            "episode_lengths": self.episode_lengths,
            "all_rewards": self.all_rewards
        }
        
        # Save as JSON
        filename = os.path.join(self.log_path, f"{self.experiment_name}_results.json")
        with open(filename, "w") as f:
            json.dump(results, f)
            
        # Generate plots
        self.plot_results(results, save_path=self.log_path)
    
    @staticmethod
    def plot_results(results, save_path="./logs"):
        """Generate and save plot visualizations of the results"""
        experiment_name = results["experiment_name"]
        timesteps = results["timesteps"]
        
        # Learning curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, results["eval_rewards"], label="Evaluation Reward")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward")
        plt.title(f"Learning Curve - {experiment_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{experiment_name}_learning_curve.png"))
        plt.close()
        
        # Success rate and exploration plot
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, results["success_rates"], label="Success Rate")
        plt.plot(timesteps, results["exploration_rates"], label="Exploration Rate")
        plt.xlabel("Timesteps")
        plt.ylabel("Rate")
        plt.title(f"Success and Exploration Rates - {experiment_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{experiment_name}_success_exploration.png"))
        plt.close()
        
        # Episode length plot
        if results["episode_lengths"]:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(results["episode_lengths"])), results["episode_lengths"])
            plt.xlabel("Episode")
            plt.ylabel("Episode Length")
            plt.title(f"Episode Lengths - {experiment_name}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"{experiment_name}_episode_length.png"))
            plt.close()


class Experiment:
    """
    Class to manage RL experiments with different combinations of 
    observation spaces, reward functions, and algorithms
    """
    def __init__(self, 
                 maps_list=None, 
                 log_dir="./logs",
                 models_dir="./models",
                 total_timesteps=1_000_000,
                 eval_freq=50_000,
                 n_eval_episodes=10):
        self.maps_list = maps_list
        self.log_dir = log_dir
        self.models_dir = models_dir
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Create directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Define observation wrappers
        self.observation_wrappers = {
            "grid": GridObservationWrapper,
            "vector": VectorObservationWrapper
        }
        
        # Define reward wrappers
        self.reward_wrappers = {
            "exploration": ExplorationRewardWrapper,
            "safety": SafetyRewardWrapper,
            "efficiency": EfficiencyRewardWrapper
        }
        
        # Define algorithms with default hyperparameters
        self.algorithms = {
            "ppo": (PPO, {
                "policy": "MlpPolicy", 
                "learning_rate": 2.5e-4,
                "n_steps": 1024,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.97,  # Focus more on immediate safety
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.05,  # Encourage exploration
                "max_grad_norm": 0.5,
                "policy_kwargs": dict(
                    net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])]
                )
            }),
            "dqn": (DQN, {
                "policy": "MlpPolicy",
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "batch_size": 64,
                "learning_starts": 1000,
                "target_update_interval": 1000,
                "exploration_fraction": 0.3,  # Increased exploration
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "policy_kwargs": dict(
                    net_arch=[128, 128]
                )
            }),
            "a2c": (A2C, {
                "policy": "MlpPolicy",
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.97,  # Focus on safety
                "ent_coef": 0.03,  # Encourage exploration
                "policy_kwargs": dict(
                    net_arch=[dict(pi=[128, 64], vf=[128, 64])]
                )
            })
        }
    
    def create_environment(self, observation_wrapper=None, reward_wrapper=None):
        """Create and wrap the environment"""
        # Create base environment
        env = gym.make(
            "standard",  # Use random maps for better generalization
            predefined_map_list=self.maps_list
        )
        
        # Add Monitor wrapper
        env = Monitor(env)
        
        # Apply observation wrapper if specified
        if observation_wrapper is not None:
            env = observation_wrapper(env)
            
        # Apply reward wrapper if specified
        if reward_wrapper is not None:
            env = reward_wrapper(env)
            
        return env
        
    def run_experiment(self, 
                       algorithm_name="ppo", 
                       observation_name="grid", 
                       reward_name="exploration"):
        """
        Run a single experiment with the specified algorithm, observation space, and reward function
        """
        # Create unique experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{algorithm_name}_{observation_name}_{reward_name}_{timestamp}"
        
        print(f"Starting experiment: {experiment_name}")
        
        # Get algorithm class and params
        algo_class, algo_params = self.algorithms[algorithm_name]
        
        # Get wrappers
        obs_wrapper = self.observation_wrappers[observation_name]
        reward_wrapper = self.reward_wrappers[reward_name]
        
        # Create training environment
        env = self.create_environment(obs_wrapper, reward_wrapper)
        
        # Create separate evaluation environment with the same configuration
        eval_env = self.create_environment(obs_wrapper, reward_wrapper)
        
        # Create callback for experiment tracking
        callback = ExperimentCallback(
            eval_env=eval_env,
            experiment_name=experiment_name,
            eval_freq=self.eval_freq,
            n_eval_episodes=self.n_eval_episodes,
            log_path=self.log_dir
        )
        
        # Initialize algorithm
        if algorithm_name == "ppo" or algorithm_name == "a2c":
            # For on-policy algorithms, use vectorized environments
            # Recreate environment as VecEnv
            env = DummyVecEnv([lambda: self.create_environment(obs_wrapper, reward_wrapper)])
            env = VecMonitor(env)
            
            # Create model
            model = algo_class(
                env=env,
                tensorboard_log=os.path.join(self.log_dir, "tensorboard"),
                **algo_params
            )
        else:
            # For off-policy algorithms like DQN
            model = algo_class(
                env=env,
                tensorboard_log=os.path.join(self.log_dir, "tensorboard"),
                **algo_params
            )
        
        # Train the model
        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callback
            )
            
            # Save the final model
            model_path = os.path.join(self.models_dir, f"{experiment_name}.zip")
            model.save(model_path)
            print(f"Model saved to {model_path}")
            
            return callback  # Return the callback for access to results
            
        except Exception as e:
            print(f"Error during training: {e}")
            return None
    
    def run_all_experiments(self, algorithm_names=None, observation_names=None, reward_names=None):
        """
        Run all combinations of experiments with the specified components
        """
        # Use defaults if not specified
        algorithm_names = algorithm_names or ["ppo"]
        observation_names = observation_names or ["grid", "vector"]
        reward_names = reward_names or ["exploration", "safety", "efficiency"]
        
        results = {}
        
        # Run all combinations
        for algo in algorithm_names:
            for obs in observation_names:
                for rew in reward_names:
                    print(f"\n{'='*50}")
                    print(f"Running experiment: {algo} + {obs} + {rew}")
                    print(f"{'='*50}\n")
                    
                    experiment_key = f"{algo}_{obs}_{rew}"
                    callback = self.run_experiment(algo, obs, rew)
                    
                    if callback:
                        results[experiment_key] = {
                            "eval_rewards": callback.eval_rewards,
                            "success_rates": callback.success_rates,
                            "exploration_rates": callback.exploration_rates
                        }
        
        # Generate comparison plots
        self.plot_comparison(results)
        
        return results
    
    def plot_comparison(self, results):
        """Generate comparison plots between different experiments"""
        if not results:
            return
            
        # Prepare figure for rewards comparison
        plt.figure(figsize=(12, 8))
        
        # Plot reward curves for each experiment
        for experiment_name, data in results.items():
            rewards = data["eval_rewards"]
            timesteps = list(range(self.eval_freq, (len(rewards) + 1) * self.eval_freq, self.eval_freq))
            plt.plot(timesteps, rewards, label=experiment_name)
        
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward")
        plt.title("Reward Comparison Across Experiments")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "reward_comparison.png"))
        plt.close()
        
        # Prepare figure for success rate comparison
        plt.figure(figsize=(12, 8))
        
        # Plot success rate curves for each experiment
        for experiment_name, data in results.items():
            success_rates = data["success_rates"]
            timesteps = list(range(self.eval_freq, (len(success_rates) + 1) * self.eval_freq, self.eval_freq))
            plt.plot(timesteps, success_rates, label=experiment_name)
        
        plt.xlabel("Timesteps")
        plt.ylabel("Success Rate")
        plt.title("Success Rate Comparison Across Experiments")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "success_rate_comparison.png"))
        plt.close()
        
        # Prepare figure for exploration rate comparison
        plt.figure(figsize=(12, 8))
        
        # Plot exploration rate curves for each experiment
        for experiment_name, data in results.items():
            exploration_rates = data["exploration_rates"]
            timesteps = list(range(self.eval_freq, (len(exploration_rates) + 1) * self.eval_freq, self.eval_freq))
            plt.plot(timesteps, exploration_rates, label=experiment_name)
        
        plt.xlabel("Timesteps")
        plt.ylabel("Exploration Rate")
        plt.title("Exploration Rate Comparison Across Experiments")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "exploration_rate_comparison.png"))
        plt.close()


def curriculum_learning(maps, start_difficulty=0, max_difficulty=4, 
                        success_threshold=0.7, algorithm_name="ppo",
                        observation_name="grid", reward_name="safety",
                        timesteps_per_level=500_000):
    """
    Implement curriculum learning by starting with easier maps
    and progressively increasing difficulty as the agent improves
    """
    # Create directories for logging and saving models
    log_dir = "./logs"
    models_dir = "./models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Get algorithm class and params
    algo_class, algo_params = {
        "ppo": (PPO, {
            "policy": "MlpPolicy", 
            "learning_rate": 2.5e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.97,  # Focus more on immediate safety
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.05,  # Encourage exploration
            "max_grad_norm": 0.5,
            "policy_kwargs": dict(
                net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])]
            )
        }),
        "dqn": (DQN, {
            "policy": "MlpPolicy",
            "learning_rate": 1e-4,
            "buffer_size": 50000,
            "batch_size": 64,
            "learning_starts": 1000,
            "target_update_interval": 1000,
            "exploration_fraction": 0.3,  # Increased exploration
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "policy_kwargs": dict(
                net_arch=[128, 128]
            )
        }),
        "a2c": (A2C, {
            "policy": "MlpPolicy",
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.97,  # Focus on safety
            "ent_coef": 0.03,  # Encourage exploration
            "policy_kwargs": dict(
                net_arch=[dict(pi=[128, 64], vf=[128, 64])]
            )
        })
    }[algorithm_name]
    
    # Get wrapper classes
    obs_wrapper_class = {
        "grid": GridObservationWrapper,
        "vector": VectorObservationWrapper
    }[observation_name]
    
    reward_wrapper_class = {
        "exploration": ExplorationRewardWrapper,
        "safety": SafetyRewardWrapper,
        "efficiency": EfficiencyRewardWrapper
    }[reward_name]
    
    # Create a model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"stealth_curriculum_{algorithm_name}_{observation_name}_{reward_name}_{timestamp}"
    
    # Start with the specified difficulty level
    current_difficulty = start_difficulty
    
    # Two-phase training approach - first focus on safety, then on exploration
    
    # Phase 1: Safety focused training
    safety_reward_wrapper = SafetyRewardWrapper
    
    def create_env_phase1():
        """Create environment with emphasis on safety"""
        env = gym.make(
            "standard",
            predefined_map_list=[maps[current_difficulty]]
        )
        env = Monitor(env)
        env = obs_wrapper_class(env)
        # Use safety wrapper with higher detection penalty
        env = safety_reward_wrapper(
            env, 
            exploration_reward=0.5,  # Lower exploration reward
            safety_reward=0.5,       # Higher safety reward
            proximity_penalty_factor=0.8,  # Stronger penalty for proximity
            detection_penalty=80.0,  # Much higher detection penalty
            completion_bonus=20.0    # Same completion bonus
        )
        return env
    
    # Create vectorized environment for PPO/A2C
    if algorithm_name in ["ppo", "a2c"]:
        env = DummyVecEnv([create_env_phase1])
        env = VecMonitor(env)
    else:
        env = create_env_phase1()
    
    # Create model
    model = algo_class(
        env=env,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        **algo_params
    )
    
    print("\n" + "="*50)
    print("PHASE 1: SAFETY TRAINING")
    print("="*50 + "\n")
    
    # Training loop with curriculum for phase 1
    for difficulty in range(start_difficulty, max_difficulty + 1):
        print(f"\n{'='*50}")
        print(f"Training on difficulty level {difficulty} - Safety Focus")
        print(f"{'='*50}\n")
        
        current_difficulty = difficulty
        
        # Recreate environment with updated difficulty
        if algorithm_name in ["ppo", "a2c"]:
            env = DummyVecEnv([create_env_phase1])
            env = VecMonitor(env)
            model.set_env(env)
        else:
            env = create_env_phase1()
            model.set_env(env)
        
        # Create eval environment for the current difficulty
        eval_env = create_env_phase1()
        
        # Train on current difficulty
        model.learn(
            total_timesteps=int(timesteps_per_level * 0.6),  # 60% of time on safety
            reset_num_timesteps=False
        )
        
        # Evaluate performance on current difficulty
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=20,
            deterministic=True
        )
        
        # Calculate success rate (no detection)
        success_count = 0
        exploration_sum = 0
        
        for _ in range(20):  # Run 20 evaluation episodes
            obs, _ = eval_env.reset()
            done = False
            detected = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                if info["game_over"]:
                    detected = True
            
            # Success for phase 1 = not being detected (regardless of exploration)
            if not detected:
                success_count += 1
                
            # Track exploration for information
            exploration_sum += info["total_covered_cells"] / info["coverable_cells"]
        
        safety_success_rate = success_count / 20
        avg_exploration = exploration_sum / 20
        
        print(f"Difficulty {difficulty}: Mean reward = {mean_reward:.2f}, "
              f"Safety success = {safety_success_rate:.2f}, "
              f"Avg exploration = {avg_exploration:.2f}")
        
        # Save model after each difficulty level
        model_path = os.path.join(models_dir, f"{experiment_name}_safety_difficulty_{difficulty}.zip")
        model.save(model_path)
        
        # If safety success rate is below threshold, repeat this level
        if safety_success_rate < success_threshold and difficulty < max_difficulty:
            print(f"Safety success rate {safety_success_rate:.2f} below threshold {success_threshold}.")
            print(f"Repeating difficulty level {difficulty}")
            difficulty -= 1  # Repeat this difficulty level
    
    # Phase 2: Balance safety with exploration
    print("\n" + "="*50)
    print("PHASE 2: BALANCED TRAINING")
    print("="*50 + "\n")
    
    # Now use the originally specified reward wrapper (exploration, safety, or efficiency)
    def create_env_phase2():
        """Create environment with balanced safety and exploration"""
        env = gym.make(
            "standard",
            predefined_map_list=[maps[current_difficulty]]
        )
        env = Monitor(env)
        env = obs_wrapper_class(env)
        env = reward_wrapper_class(env)  # Use the specified reward wrapper
        return env
    
    # Training loop with curriculum for phase 2
    for difficulty in range(start_difficulty, max_difficulty + 1):
        print(f"\n{'='*50}")
        print(f"Training on difficulty level {difficulty} - Balanced Focus")
        print(f"{'='*50}\n")
        
        current_difficulty = difficulty
        
        # Recreate environment with updated difficulty
        if algorithm_name in ["ppo", "a2c"]:
            env = DummyVecEnv([create_env_phase2])
            env = VecMonitor(env)
            model.set_env(env)
        else:
            env = create_env_phase2()
            model.set_env(env)
        
        # Create eval environment for the current difficulty
        eval_env = create_env_phase2()
        
        # Train on current difficulty
        model.learn(
            total_timesteps=int(timesteps_per_level * 0.4),  # 40% of time on balanced training
            reset_num_timesteps=False
        )
        
        # Evaluate performance on current difficulty
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=20,
            deterministic=True
        )
        
        # Calculate success rate and exploration rate
        success_count = 0
        full_success_count = 0
        exploration_sum = 0
        
        for _ in range(20):  # Run 20 evaluation episodes
            obs, _ = eval_env.reset()
            done = False
            detected = False
            final_info = None
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                if done:
                    final_info = info
                if info["game_over"]:
                    detected = True
            
            total_covered = final_info["total_covered_cells"]
            coverable_cells = final_info["coverable_cells"]
            
            # Success = not being detected
            if not detected:
                success_count += 1
                
                # Full success = not detected AND explored all cells
                if total_covered == coverable_cells:
                    full_success_count += 1
                
            # Track exploration percentage
            exploration_sum += total_covered / coverable_cells
        
        safety_success_rate = success_count / 20
        full_success_rate = full_success_count / 20
        avg_exploration = exploration_sum / 20
        
        print(f"Difficulty {difficulty}: Mean reward = {mean_reward:.2f}, "
              f"Safety success = {safety_success_rate:.2f}, "
              f"Full success = {full_success_rate:.2f}, "
              f"Avg exploration = {avg_exploration:.2f}")
        
        # Save model after each difficulty level
        model_path = os.path.join(models_dir, f"{experiment_name}_balanced_difficulty_{difficulty}.zip")
        model.save(model_path)
        
        # Only repeat if both safety AND exploration are below threshold
        if (safety_success_rate < success_threshold or avg_exploration < 0.7) and difficulty < max_difficulty:
            print(f"Performance below thresholds. Repeating difficulty level {difficulty}")
            difficulty -= 1  # Repeat this difficulty level
    
    # Final model save
    final_model_path = os.path.join(models_dir, f"{experiment_name}_final.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model, final_model_path


# Function to run the agent with render mode for visualization
def visualize_agent(model_path, map_index=4, map_name=None, max_episodes=3, render_mode="human"):
    """
    Visualize trained agent performance on a specific map
    
    Args:
        model_path: Path to the trained model
        map_index: Index of the map to use (used if map_name is None)
        map_name: Name of the map to use (takes precedence over map_index)
        max_episodes: Number of episodes to run
        render_mode: Rendering mode
    """
    from gymnasium.wrappers import TimeLimit
    
    # Define predefined map names for easy reference
    PREDEFINED_MAPS = [
        "just_go",        # very easy - no obstacles
        "safe",           # easy - some walls
        "maze",           # medium - maze with enemies
        "chokepoint",     # hard - narrow passages with enemies
        "sneaky_enemies"  # very hard - many enemies in open space
    ]
    
    # Load the model
    model_type = model_path.split('_')[0]
    
    if "ppo" in model_type:
        model = PPO.load(model_path)
    elif "dqn" in model_type:
        model = DQN.load(model_path)
    elif "a2c" in model_type:
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Extract the observation and reward wrapper types from the path
    parts = os.path.basename(model_path).split('_')
    observation_type = parts[1] if len(parts) > 1 else "grid"
    reward_type = parts[2] if len(parts) > 2 else "exploration"
    
    # Use either map_name (preferred) or map_index
    if map_name is not None:
        # Use the provided map name
        env_id = map_name
    else:
        # Use index to get map name
        if 0 <= map_index < len(PREDEFINED_MAPS):
            env_id = PREDEFINED_MAPS[map_index]
        else:
            # Default to hardest map if index is out of range
            env_id = "sneaky_enemies"
    
    # Create environment with the same wrappers used during training
    env = gym.make(
        env_id, 
        render_mode=render_mode,
        activate_game_status=True
    )
    
    # Apply wrappers
    obs_wrapper = {
        "grid": GridObservationWrapper,
        "vector": VectorObservationWrapper
    }[observation_type]
    
    reward_wrapper = {
        "exploration": ExplorationRewardWrapper,
        "safety": SafetyRewardWrapper,
        "efficiency": EfficiencyRewardWrapper
    }[reward_type]
    
    env = obs_wrapper(env)
    env = reward_wrapper(env)
    
    # Run episodes
    for episode in range(max_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        print(f"Episode {episode+1}/{max_episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            
            # Slow down visualization
            time.sleep(0.05)
        
        # Print episode summary
        print(f"Episode {episode+1} finished with reward {total_reward:.2f}")
        print(f"Steps taken: {step}")
        print(f"Cells covered: {info['total_covered_cells']}/{info['coverable_cells']}")
        print(f"Game over by enemy detection: {info['game_over']}")
        print("-" * 50)
        
        # Pause between episodes
        time.sleep(1)
    
    env.close()
    
    return env


# Main execution code
if __name__ == "__main__":
    from main import maps
    
    # Run our stealth-focused curriculum learning
    model, model_path = curriculum_learning(
        maps=maps,
        start_difficulty=0,
        max_difficulty=4,
        algorithm_name="ppo",
        observation_name="grid",
        reward_name="safety",
        timesteps_per_level=200_000  # Shorter for faster testing
    )
    
    # Visualize the trained agent
    visualize_agent(
        model_path=model_path,
        map_index=4,  # Test on hardest map (sneaky_enemies)
        max_episodes=3
    )

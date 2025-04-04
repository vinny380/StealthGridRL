import os
import random
import sys
import time
import gymnasium as gym

from stable_baselines3 import DQN

# Add coverage-gridworld to path
sys.path.append("D:\\Documents\\CISC474_Project\\StealthGridRL\\coverage-gridworld")
print(sys.path)

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

for i in range(num_episodes):
    env.reset()
    # obs = env.get_state()
    done = False
    # Do nothing for one step, just to get danger table values
    obs, reward, done, truncated, info = env.step(4)
    danger_table = determine_cell_danger_tables(env, info["enemies"])
    custom.set_danger_table(danger_table)
    time.sleep(1)
    while done is False:
        custom.incr_timestep()
        action = rl_player()
        obs, reward, done, truncated, info = env.step(action)
        # Sleep may be used to allow each step to be visualized. Value can be changed
        #time.sleep(0.2)
    if done:
        time.sleep(2)
env.close()

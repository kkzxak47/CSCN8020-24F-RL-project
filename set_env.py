import gymnasium as gym
import sys
import os
sys.path.append(os.getcwd())
from custom_envs.custom_highway_env import CustomHighwayEnv


env = gym.make('CustomHighway-v0', render_mode='rgb_array')
# env = gym.make('highway-v0', render_mode='rgb_array')

env.unwrapped.configure({
    "duration": 60,  # Total steps in the episode
    "vehicles_count": 20,
    "lanes_count": 4,  # Number of lanes
    "reward_speed_range": [20, 30],  # Speed range for rewards
    "policy_frequency": 2,  # Action frequency
    "simulation_frequency": 15,  # Simulation frequency
    "collision_reward": -1,  # Collision reward
    "right_lane_reward": 0,  # Reward for driving on the rightmost lanes
    "lane_change_reward": 0.01,  # Reward for lane changes
    # "ego_spacing": 2.0,  # Desired distance between the ego vehicle and the leading vehicle
    "controlled_vehicle": {
        "type": "highway_env.vehicle.behavior.IDMVehicle",
        "parameters": {"max_speed": 30}  # Increase max speed
    },
    # 'other_vehicles_type': 'highway_env.vehicle.behavior.DefensiveVehicle',
    "high_speed_reward": 0.75,    # Reward for maintaining high speed
    "distance_reward": 0.05,  # 0.5 is too high, tweak it
    "initial_spacing": 2,  # Initial spacing between vehicles
    "terminal_conditions": ["off_road", "time_limit"],
    "vehicles_density": 1,
    "offroad_terminal": True,
    # "show_trajectories": True,
    "road_length": 1800,  # set this to a large number to avoid road length limit, also 30 * 60 = 1800, this is the max_speed * duration
})
print(env.unwrapped.config)
env.reset()

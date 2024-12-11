import gymnasium as gym
import sys
import os
sys.path.append(os.getcwd())
from custom_envs.custom_highway_env import CustomHighwayEnv, ROAD_LENGTH


env = gym.make('CustomHighway-v0', render_mode='rgb_array')
# env = gym.make('highway-v0', render_mode='rgb_array')

env.unwrapped.configure({
    "duration": 60,  # Time limit [s]
    "vehicles_count": 20,
    "lanes_count": 4,  # Number of lanes
    "reward_speed_range": [20, 30],  # Speed range for rewards
    "policy_frequency": 5,  # Action frequency
    "simulation_frequency": 15,  # Simulation frequency
    "collision_reward": -1,  # Collision reward
    "right_lane_reward": 0,  # Reward for driving on the rightmost lanes
    "lane_change_reward": 0.01,  # Reward for lane changes
    # "ego_spacing": 3,  # Desired distance between the ego vehicle and the leading vehicle
    "controlled_vehicle": {
        "type": "highway_env.vehicle.behavior.IDMVehicle",
        "parameters": {"max_speed": 30}  # Increase max speed
    },
    # 'other_vehicles_type': 'highway_env.vehicle.behavior.DefensiveVehicle',
    "high_speed_reward": 0.25,    # Reward for maintaining high speed
    "distance_reward": 0.25,  # 0.5 is too high, tweak it
    "initial_spacing": 2,  # Initial spacing between vehicles
    "terminal_conditions": ["off_road", "time_limit"],
    "vehicles_density": 1,
    "offroad_terminal": True,
    # "show_trajectories": True,
    "road_length": ROAD_LENGTH,
})
print(env.unwrapped.config)
env.reset()

print(f"{env.unwrapped.initial_position=}")

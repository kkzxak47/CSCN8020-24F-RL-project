import gymnasium as gym
from highway_env.envs import HighwayEnv

# class CustomHighwayEnv(HighwayEnv):
#     def _reward(self, action):
#         vehicle = self.vehicle
#         speed = vehicle.speed
#         max_speed = vehicle.MAX_SPEED

#         # Reward based on speed
#         speed_reward = speed / max_speed

#         # Penalize collisions
#         collision_penalty = -1 if self.vehicle.crashed else 0

#         # Penalize low speed
#         low_speed_penalty = -0.1 if speed < max_speed / 2 else 0

#         # Reward efficient lane changes
#         lane_change_reward = 0.1 if action in [0, 2] else 0

#         return speed_reward + collision_penalty + low_speed_penalty + lane_change_reward

env = gym.make('highway-v0', render_mode='rgb_array')

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
    "high_speed_reward": 0.5,    # Reward for maintaining high speed
    "initial_spacing": 2,  # Initial spacing between vehicles
    "terminal_conditions": ["off_road", "time_limit"],
    "vehicles_density": 1,
    "offroad_terminal": True,
    # "show_trajectories": True,
})
print(env.unwrapped.config)
env.reset()
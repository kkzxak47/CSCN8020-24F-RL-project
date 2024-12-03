import gymnasium as gym
import highway_env  # Ensure highway-env is imported


def create_environment():
    """
    Creates and configures the highway-v0 environment.
    """
    env = gym.make("highway-v0", render_mode="rgb_array")  # Use pre-built scenario
    env.unwrapped.configure(  # Apply configuration to the unwrapped environment
        {
            "duration": 60,  # Total steps in an episode
            "vehicles_count": 20,  # Number of vehicles
            "lanes_count": 4,  # Number of lanes
            "reward_speed_range": [20, 30],  # Speed range for rewards
            "policy_frequency": 2,  # Frequency of agent actions
            "simulation_frequency": 15,  # Frequency of simulation steps
            "collision_reward": -1,  # Penalty for collisions
            "right_lane_reward": 0.1,  # Reward for staying in the right lane
            "high_speed_reward": 0.5,  # Reward for high speed
            "distance_reward": 0.1,  # Reward for distance traveled
            "road_length": 1800,  # Length of the road
            "offroad_terminal": True,  # Terminate if agent leaves the road
        }
    )
    return env

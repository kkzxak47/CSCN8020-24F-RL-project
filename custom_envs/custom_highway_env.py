from __future__ import annotations

import logging
import gymnasium as gym

from highway_env.envs import HighwayEnv


import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

# Register the custom environment
from gymnasium.envs.registration import register
register(
    id="CustomHighway-v0",  # Custom environment ID
    entry_point="custom_envs.custom_highway_env:CustomHighwayEnv",  # Replace with the actual path to the class
)

print('CustomHighway-v0 registered')
assert 'CustomHighway-v0' in gym.envs.registry.keys()
# print(gym.envs.registry.keys())
logger = logging.getLogger(__name__)


ROAD_LENGTH = 90000000  # long road, no way to reach the end

Observation = np.ndarray

class CustomHighwayEnv(HighwayEnv):
    def __init__(self, config: dict = None, render_mode: str | None = None) -> None:
        super().__init__(config=config, render_mode=render_mode)
        self.initial_position = 0
    
    def _reset(self) -> None:
        logger.info("Resetting environment, customized _reset, set initial_position to controlled vehicle position")
        super()._reset()
        self.initial_position = self.vehicle.position[0]

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["distance_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "distance_reward": (self.vehicle.position[0] - self.initial_position) / ROAD_LENGTH,
        }

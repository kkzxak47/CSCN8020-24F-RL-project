import sys
import os
sys.path.append(os.getcwd())  # Add the current directory to sys.path
import gymnasium as gym
# import highway_env
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from gymnasium.wrappers import RecordVideo


import custom_envs.custom_highway_env as custom_highway_env
from custom_envs.custom_highway_env import CustomHighwayEnv
import sys


import sys
if len(sys.argv) > 1:
    step = sys.argv[1]
    assert step.isnumeric() and int(step) > 0 and int(step) % 1000 == 0, "Invalid step number. Please provide a valid step number."
    DQN_MODEL_PATH = f"./checkpoints/model_step_{step}.zip"
else:
    DQN_MODEL_PATH = "highway_dqn.model.bin"
    # DQN_MODEL_PATH = "checkpoints-dqn-342k/model_step_342000.zip"
# load the model
model = DQN.load(DQN_MODEL_PATH)
# model = PPO.load("highway_ppo.model.bin")

from set_env import env
env.unwrapped.render_mode = "human"
# env = RecordVideo(env, video_folder='videos', episode_trigger=lambda e: True)
env.reset()

distances = []
for episode in range(5):
    (obs, info), done, truncated = env.reset(), False, False
    print(f"Episode {episode + 1}")
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
    distances.append(env.unwrapped.vehicle.position[0] - env.unwrapped.initial_position)
print(f"Mean distance: {np.mean(distances)}")
print(f"Max distance: {np.max(distances)}")
print(f"Min distance: {np.min(distances)}")
env.close()

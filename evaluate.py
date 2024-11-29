import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from gymnasium.wrappers import RecordVideo

import sys
step = sys.argv[1]
assert step.isnumeric() and int(step) > 0 and int(step) % 1000 == 0, "Invalid step number. Please provide a valid step number."
# DQN_MODEL_PATH = "highway_dqn.model.bin"
DQN_MODEL_PATH = f"./checkpoints/model_step_{step}.zip"
# load the model
model = DQN.load(DQN_MODEL_PATH)
# model = PPO.load("highway_ppo.model.bin")

from set_env import env
env.unwrapped.render_mode = "human"
# env = RecordVideo(env, video_folder='videos', episode_trigger=lambda e: True)
env.reset()

for episode in range(3):    
    (obs, info), done, truncated = env.reset(), False, False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
env.close()

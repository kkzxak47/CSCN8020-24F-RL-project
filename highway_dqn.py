
# # Highway with SB3's DQN
# 
# ##  Warming up
# We start with a few useful installs and imports:


# Install environment and agent
# %pip install highway-env
# # TODO: we use the bleeding edge version because the current stable version does not support the latest gym>=0.21 versions. Revert back to stable at the next SB3 release.
# %pip install git+https://github.com/DLR-RM/stable-baselines3
# %pip install tensorboardx gym pyvirtualdisplay tqdm


# %load_ext dotenv
# %dotenv

# Environment
from typing import Dict
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)


# Agent
from stable_baselines3 import DQN, PPO


# Visualization utils
# %load_ext tensorboard
import sys
from tqdm.notebook import trange

# !git clone https://github.com/Farama-Foundation/HighwayEnv.git 2> /dev/null
# sys.path.insert(0, './HighwayEnv/scripts/')
# from HighwayEnv.scripts.utils import record_videos, show_videos

DQN_MODEL_FILE = "highway_dqn.model.bin"

# ## Training
# Run tensorboard locally to visualize training.


# %tensorboard --logdir "highway_dqn"


# env = gym.make('highway-v0', render_mode='rgb_array')
# env.unwrapped.configure({
#     "duration": 60,  # Total steps in the episode
#     "vehicles_count": 20,  # Traffic density
#     "lanes_count": 4,  # Number of lanes
#     "reward_speed_range": [20, 30],  # Speed range for rewards
#     "policy_frequency": 2,  # Action frequency
#     "simulation_frequency": 24,  # Simulation frequency
#     "collision_reward": -1.0,  # Collision reward
#     "right_lane_reward": 0,  # Reward for driving on the rightmost lanes
#     "lane_change_reward": 0.1,  # Reward for lane changes
#     "speed_reward": 1.0,  # Speed reward
# })
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import os

from torch.optim.lr_scheduler import LambdaLR

# dynamic learning rate schedule
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# scheduler = LambdaLR(optimizer, lr_lambda=lambda t: 1 / (1 + 1e-5 * t))

# Define a custom learning rate schedule
def lr_schedule(progress_remaining):
    return 0.0009 * progress_remaining + 0.0001


from model_checkpoint_callback import callback

from set_env import env
# env.unwrapped.render_mode = "human"
# env.reset()
if os.path.exists(DQN_MODEL_FILE):
    print(f"Loading model from {DQN_MODEL_FILE}")
    model = DQN.load(DQN_MODEL_FILE, env)
else:
    print("Training a new model.")
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[64, 128, 64]),
                # learning_rate=5e-4,
                learning_rate=lr_schedule,
                buffer_size=15000,
                learning_starts=8000,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.5,
                verbose=1,
                tensorboard_log='highway_dqn/')

try:
    print("Press Ctrl+C to interrupt training. ")
    model.learn(int(1e6), callback=callback)  # 1 million steps
except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected! Cleaning up and exiting gracefully.")        


model.save(DQN_MODEL_FILE)



# ## Testing
# 
# Visualize a few episodes


# load the model
model = DQN.load(DQN_MODEL_FILE)


# env = gym.make('highway-fast-v0', render_mode='rgb_array')
# env = gym.make('highway-v0', render_mode='rgb_array')

# env.reset()
from gymnasium.wrappers import RecordVideo
env = RecordVideo(env, video_folder='videos', episode_trigger=lambda e: True)
# env = record_videos(env)
for episode in trange(3, desc='Test episodes'):
    (obs, info), done, truncated = env.reset(), False, False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        # env.render()
env.close()
# show_videos()



# from stable_baselines3 import PPO

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='highway_ppo/')
# model.learn(int(5e4))
# model.save("highway_ppo.model.bin")




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
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)

# Agent
from stable_baselines3 import DQN


# Visualization utils
# %load_ext tensorboard
import sys
from tqdm.notebook import trange

# !git clone https://github.com/Farama-Foundation/HighwayEnv.git 2> /dev/null
# sys.path.insert(0, './HighwayEnv/scripts/')
from HighwayEnv.scripts.utils import record_videos, show_videos

DQN_MODEL_FILE = "highway_dqn.model.bin"

# ## Training
# Run tensorboard locally to visualize training.


# %tensorboard --logdir "highway_dqn"


# env = gym.make('highway-v0', render_mode='rgb_array')
# env.unwrapped.configure({
#     "duration": 1000,  # Total steps in the episode
#     "vehicles_count": 50,  # Traffic density
#     "lanes_count": 4,  # Number of lanes
#     "reward_speed_range": [20, 30],  # Speed range for rewards
#     "policy_frequency": 2,  # Action frequency
#     "simulation_frequency": 24,  # Simulation frequency
#     "collision_reward": -10,  # Collision reward
#     "right_lane_reward": 0,  # Reward for driving on the rightmost lanes
#     "lane_change_reward": 0,  # Reward for lane changes
#     "speed_reward": 10,  # Speed reward
# })
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import os

# Create a callback to save the model periodically
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"model_step_{self.n_calls}")
            self.model.save(model_path)
            if self.verbose:
                print(f"Model saved to {model_path}")
        return True

# Save directory
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)
# Train the model and save periodically
save_freq = 2000  # Save every 2000 steps
callback = SaveOnBestTrainingRewardCallback(save_freq=save_freq, save_path=save_dir)

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
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log='highway_dqn/')
try:
    print("Press Ctrl+C to interrupt training. ")
    # model.learn(int(2e5), callback=callback)
except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected! Cleaning up and exiting gracefully.")        

# model.save(DQN_MODEL_FILE)



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
    (obs, info), done = env.reset(), False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        # env.render()
env.close()
show_videos()



# from stable_baselines3 import PPO

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='highway_ppo/')
# model.learn(int(5e4))
# model.save("highway_ppo.model.bin")



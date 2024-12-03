from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import RecordVideo
from env.custom_env import create_environment
import numpy as np
import os

# Directory for saving evaluation videos
video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)

# Create and wrap the environment for video recording
env = create_environment()
env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda e: True, name_prefix="ppo_evaluation")

# Vectorize the environment for consistency with training
vec_env = make_vec_env(lambda: env, n_envs=1)

# Load the trained PPO model
model = PPO.load("./ppo_highway_model_final", env=vec_env)
print("Model loaded successfully.")

# Evaluate the model
n_episodes = 5  # Number of episodes to evaluate
print(f"Starting evaluation for {n_episodes} episodes...")

episode_rewards = []
for episode in range(n_episodes):
    obs = vec_env.reset()  # Reset the environment
    done = False
    total_reward = 0
    while not done:
        # Get the action from the trained model
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)

        # Update total reward and handle VecEnv response
        total_reward += rewards[0]  # VecEnv returns a list of rewards
        done = dones[0]  # VecEnv returns a list of done flags

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")

# Calculate and display summary statistics
avg_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
print(f"Evaluation completed: Avg Reward: {avg_reward:.2f}, Std Reward: {std_reward:.2f}")

# Close the environment
vec_env.close()
print(f"Videos saved to: {video_dir}")

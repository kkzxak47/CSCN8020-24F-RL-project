import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env.custom_env import create_environment

# Create the environment
env = create_environment()
vec_env = make_vec_env(lambda: env, n_envs=1)

# Initialize PPO model
model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./logs/ppo_highway/",
)

# Checkpoint callback to save models periodically
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # Save every 10k timesteps
    save_path="./models/",
    name_prefix="ppo_checkpoint",
)

# Train the model
print("Training started...")
model.learn(total_timesteps=100000, callback=checkpoint_callback)
model.save("ppo_highway_model_final")
print("Training completed! Final model saved as 'ppo_highway_model_final'.")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from set_env import env
from model_checkpoint_callback import callback


# Vectorize the environment for PPO
vec_env = make_vec_env(lambda: env, n_envs=1)
# Initialize the PPO model
# model = PPO(
#     "MlpPolicy",      # Multi-layer perceptron policy
#     vec_env,          # Vectorized environment
#     verbose=1,        # Show training details
#     tensorboard_log="./ppo_highway_tensorboard/"  # Log for TensorBoard
# )
n_cpu = 6
batch_size = 64
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
    n_steps=batch_size * 12 // n_cpu,
    batch_size=batch_size,
    n_epochs=10,
    learning_rate=5e-4,
    gamma=0.9,
    verbose=2,
    tensorboard_log="highway_ppo/",
    
)

# Train the model
model.learn(total_timesteps=3e5, callback=callback)  # Adjust timesteps as needed

# Save the model
model.save("ppo_highway_model-3e5")

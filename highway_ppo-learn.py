from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from set_env import env



# Vectorize the environment for PPO
vec_env = make_vec_env(lambda: env, n_envs=1)
# Initialize the PPO model
model = PPO(
    "MlpPolicy",      # Multi-layer perceptron policy
    vec_env,          # Vectorized environment
    verbose=1,        # Show training details
    tensorboard_log="./ppo_highway_tensorboard/"  # Log for TensorBoard
)

# Train the model
model.learn(total_timesteps=300000)  # Adjust timesteps as needed

# Save the model
model.save("ppo_highway_model")

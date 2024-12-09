from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
import os

from model_checkpoint_callback import callback
from set_env import env

A2C_MODEL_FILE = "highway_a2c.model.bin"

# Define a custom policy network architecture
policy_kwargs = dict(net_arch=[dict(pi=[64, 128, 64], vf=[64, 128, 64])])

# Load or create an A2C model
if os.path.exists(A2C_MODEL_FILE):
    print(f"Loading model from {A2C_MODEL_FILE}")
    model = A2C.load(A2C_MODEL_FILE, env)
else:
    print("Training a new model.")
    model = A2C(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=7e-4,  # Default learning rate for A2C
        gamma=0.99,  # Discount factor
        n_steps=5,  # Number of steps to roll out before updating
        vf_coef=0.5,  # Value function loss coefficient
        ent_coef=0.01,  # Entropy term coefficient (exploration encouragement)
        max_grad_norm=0.5,  # Gradient clipping
        tensorboard_log="highway_a2c/",  # TensorBoard log directory
        verbose=1,
    )

try:
    print("Press Ctrl+C to interrupt training.")
    model.learn(total_timesteps=int(1e6), callback=callback)  # Train for 1 million steps
except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected! Cleaning up and exiting gracefully.")

# Save the trained model
model.save(A2C_MODEL_FILE)
print(f"Model saved to {A2C_MODEL_FILE}")

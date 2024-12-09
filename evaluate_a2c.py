from stable_baselines3 import A2C
from set_env import env
import numpy as np

A2C_MODEL_PATH = "highway_a2c.model.bin"

# Load the A2C model
model = A2C.load(A2C_MODEL_PATH)

# Evaluate the model
distances = []
for episode in range(3):  # Evaluate for 3 episodes
    (obs, info), done, truncated = env.reset(), False, False
    print(f"Episode {episode + 1}")
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
    distances.append(env.unwrapped.vehicle.position[0] - env.unwrapped.initial_position)

# Print evaluation results
print(f"Mean distance: {np.mean(distances)}")
print(f"Max distance: {np.max(distances)}")
print(f"Min distance: {np.min(distances)}")
env.close()

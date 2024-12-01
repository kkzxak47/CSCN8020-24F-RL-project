from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import RecordVideo
from set_env import env

env.unwrapped.render_mode = "human"
env.reset()
# env = RecordVideo(env, video_folder='videos', episode_trigger=lambda e: True)

# Vectorize the environment for PPO
vec_env = make_vec_env(lambda: env, n_envs=1)

import sys
step = sys.argv[1]
assert step.isnumeric() and int(step) > 0 and int(step) % 1000 == 0, "Invalid step number. Please provide a valid step number."
# DQN_MODEL_PATH = "highway_dqn.model.bin"
DQN_MODEL_PATH = f"./checkpoints/model_step_{step}.zip"
# load the model

# model = PPO.load("highway_ppo.model.bin")
# Load the trained model
model = PPO.load(DQN_MODEL_PATH, env=vec_env)
obs = vec_env.reset()

for episode in range(3):
    obs = vec_env.reset()
    env = vec_env.envs[0]
    print(f"Episode {episode + 1}")
    
    done = False
    position = 0
    speed = 0
    while not (done):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        prev_position = position
        prev_speed = speed
        # Store position and speed before rendering
        position = env.unwrapped.vehicle.position[0]
        speed = env.unwrapped.vehicle.speed
        print(position, speed)
        # print(env.unwrapped.vehicle.position[0], env.unwrapped.vehicle.speed)
        vec_env.render()  # Use vec_env for rendering
        done = dones[0]
        if done:
            print(f"Final distance: {prev_position} speed {prev_speed}")   
    
    
    
vec_env.close()

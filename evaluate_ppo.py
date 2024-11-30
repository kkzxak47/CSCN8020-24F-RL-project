from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import RecordVideo
from set_env import env

# env.unwrapped.render_mode = "human"
# env.reset()
env = RecordVideo(env, video_folder='videos', episode_trigger=lambda e: True)

# Vectorize the environment for PPO
vec_env = make_vec_env(lambda: env, n_envs=1)


# Load the trained model
model = PPO.load("ppo_highway_model", env=vec_env)
obs = vec_env.reset()

for episode in range(2):
    obs = vec_env.reset()
    env = vec_env.envs[0]
    print(f"Episode {episode + 1}")
    
    done = False
    while not (done):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        print(env.unwrapped.vehicle.position[0], env.unwrapped.vehicle.speed)
        vec_env.render()  # Use vec_env for rendering
        done = dones[0]
    
    
    print(f"Final distance: {env.unwrapped.vehicle.position[0]}")   
vec_env.close()

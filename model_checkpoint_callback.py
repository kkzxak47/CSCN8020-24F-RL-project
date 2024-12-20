import os
from stable_baselines3.common.callbacks import BaseCallback


# Create a callback to save the model periodically
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_mean_reward = -float("inf")
    
    def _on_rollout_end(self) -> None:
        env = self.training_env.envs[0].unwrapped
        self.logger.record("rollout/distance", env.vehicle.position[0] - env.initial_position)
        return super()._on_rollout_end()
    
    # def _on_training_start(self) -> None:
    #     print("*** on training start")
    #     print(self.num_timesteps)
    #     print(f"{self.training_env.envs[0].unwrapped.steps}")

    # def _on_training_end(self) -> None:
    #     print("*** on training end")
    #     print(self.num_timesteps)
    #     print(f"{self.training_env.envs[0].unwrapped.steps}")

    def _on_step(self) -> bool:
        # print speed, find position
        
        # print(env.unwrapped.vehicle.speed, env.unwrapped.vehicle.position)
        
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

import gym
import numpy as np
import airsim
import time
import math
import os
import sys
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch
from tqdm import tqdm


# ----- Custom Callback to log step counts and episodes with progress bar -----
class ProgressBarCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.episode_count = 0
        self.best_reward = -float('inf')
        self.pbar = None
        self.total_episodes = 100
        self.current_episode_steps = 0
        self.max_steps_per_episode = 200

    def _on_training_start(self) -> None:
        # Initialize progress bar for episodes
        self.pbar = tqdm(total=self.total_episodes, desc="Episodes", position=0, leave=True)

    def _on_step(self) -> bool:
        # Check if we've reached the target number of episodes
        if hasattr(self.model.get_env(), "venv"):
            env = self.model.get_env().venv.envs[0]
            
            # Update step counter for current episode
            if hasattr(env, "steps_taken") and env.steps_taken != self.current_episode_steps:
                self.current_episode_steps = env.steps_taken
                # Update progress bar description with current step info
                if hasattr(env, "current_distance") and hasattr(env, "battery_percentage") and hasattr(env, "episode_reward"):
                    self.pbar.set_postfix({
                        "Steps": f"{env.steps_taken}/{self.max_steps_per_episode}",
                        "Distance": f"{env.current_distance:.2f}",
                        "Battery": f"{env.battery_percentage:.1f}%",
                        "Reward": f"{env.episode_reward:.1f}"
                    })
            
            # Check if episode ended
            if hasattr(env, "episode_complete") and env.episode_complete:
                self.episode_count += 1
                reward = env.episode_reward
                env.episode_complete = False
                
                # Update progress bar
                self.pbar.update(1)
                self.pbar.set_postfix({
                    "Episode": f"{self.episode_count}/{self.total_episodes}",
                    "Reward": f"{reward:.1f}",
                    "Best": f"{self.best_reward:.1f}"
                })
                
                # Reset step counter for next episode
                self.current_episode_steps = 0
                
                # Save best model
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.model.save("best_drone_model")
                    print(f"\n[Callback] New best model saved with reward: {reward:.2f}")
                
                if self.episode_count >= self.total_episodes:
                    print("\n[Callback] Reached 100 episodes, stopping training.")
                    self.pbar.close()
                    return False
                    
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


# ----- Define the custom Gym environment -----
class DroneEnv(gym.Env):
    """A custom Gym environment for training an RL agent to control a drone in AirSim."""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(DroneEnv, self).__init__()

        # Define the action space: 4 continuous actions [vx, vy, vz, yaw_rate]
        self.action_space = spaces.Box(low=np.array([-5, -5, -2, -45], dtype=np.float32),
                                       high=np.array([5, 5, 2, 45], dtype=np.float32),
                                       dtype=np.float32)

        # Observation: relative position of the drone to target: [dx, dy, dz, battery]
        self.observation_space = spaces.Box(low=np.array([-100, -100, -100, 0], dtype=np.float32),
                                           high=np.array([100, 100, 100, 100], dtype=np.float32),
                                           dtype=np.float32)

        # Connect to AirSim.
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("[DroneEnv] Connected to AirSim.")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Fixed start and target points (NED coordinates).
        self.initial_position = airsim.Vector3r(0.0, 0.0, -0.6777)  # Starting position
        self.target_position = airsim.Vector3r(-40.972, 0.878, -0.676)  # Target position
        self.target_threshold = 0.5  # Barrier of 0.5 on each axis

        # Battery management
        self.battery_percentage = 100.0
        self.battery_drain_per_action = 0.5  # 1% drop per 2 actions
        self.max_steps = 200  # Maximum steps per episode

        # Episode tracking
        self.episode_reward = 0
        self.steps_taken = 0
        self.episode_complete = False
        self.episode_num = 0
        self.collision_occurred = False
        
        self.duration = 0.5  # Duration for each action command.
        self.prev_distance = None
        self.current_distance = 0

    def reset(self):
        self.episode_num += 1
        print(f"\n[Episode {self.episode_num}/100] Starting new episode")
        self.client.reset()
        time.sleep(0.2)  # Reduced wait time for faster training

        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Reset episode variables
        self.battery_percentage = 100.0
        self.episode_reward = 500  # Initialize reward to 500
        self.steps_taken = 0
        self.episode_complete = False
        self.collision_occurred = False

        # Take off.
        self.client.takeoffAsync().join()
        time.sleep(0.2)  # Reduced wait time

        # Move the drone to the initial position.
        self.client.moveToPositionAsync(
            self.initial_position.x_val, 
            self.initial_position.y_val, 
            self.initial_position.z_val, 
            5
        ).join()
        time.sleep(0.2)  # Reduced wait time

        # Plot the target marker in the environment.
        # The target point will be a large green marker.
        self.client.simPlotPoints(
            [self.target_position], 
            color_rgba=[0, 1, 0, 1], 
            size=25.0, 
            duration=60,
            is_persistent=True
        )

        # Get initial observation.
        pose = self.client.simGetVehiclePose().position
        dx = self.target_position.x_val - pose.x_val
        dy = self.target_position.y_val - pose.y_val
        dz = self.target_position.z_val - pose.z_val
        
        # Calculate initial distance to target
        self.current_distance = np.sqrt(dx**2 + dy**2 + dz**2)
        self.prev_distance = self.current_distance
        
        print(f"[Episode {self.episode_num}] Initial position: ({pose.x_val:.3f}, {pose.y_val:.3f}, {pose.z_val:.3f})")
        print(f"[Episode {self.episode_num}] Distance to target: {self.current_distance:.3f}")
        print(f"[Episode {self.episode_num}] Battery: {self.battery_percentage:.1f}%")
        print(f"[Episode {self.episode_num}] Initial reward: {self.episode_reward}")
        
        # Return observation: [dx, dy, dz, battery_percentage]
        obs = np.array([dx, dy, dz, self.battery_percentage], dtype=np.float32)
        return obs

    def step(self, action):
        # Unpack action.
        vx, vy, vz, yaw_rate = action

        # Cast to Python floats to avoid msgpack errors.
        vx = float(vx)
        vy = float(vy)
        vz = float(vz)
        yaw_rate = float(yaw_rate)
        duration = float(self.duration)

        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)

        # Send command.
        self.client.moveByVelocityAsync(vx, vy, vz, duration, yaw_mode=yaw_mode).join()
        time.sleep(0.05)  # Balanced sleep time for performance and stability

        # Update step count and battery
        self.steps_taken += 1
        self.battery_percentage -= self.battery_drain_per_action
        self.battery_percentage = max(0, self.battery_percentage)

        # Get new observation.
        pose = self.client.simGetVehiclePose().position
        dx = self.target_position.x_val - pose.x_val
        dy = self.target_position.y_val - pose.y_val
        dz = self.target_position.z_val - pose.z_val
        
        # Calculate current distance to target
        self.current_distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Determine reward based on distance change
        step_reward = 0
        if self.current_distance < self.prev_distance:
            step_reward = 1  # Distance decreased
        else:
            step_reward = -1  # Distance increased

        # Check for collision
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided and not self.collision_occurred:
            step_reward = -20
            self.collision_occurred = True
            print(f"\n[EVENT] Collision occurred at step {self.steps_taken}!")
            # Note: We don't end the episode on collision as per requirements

        # Update episode reward
        self.episode_reward += step_reward
        
        # Check if episode should end
        done = False
        
        # Check if target reached (within threshold on all axes)
        target_reached = (
            abs(dx) <= self.target_threshold and
            abs(dy) <= self.target_threshold and
            abs(dz) <= self.target_threshold
        )
        
        if target_reached:
            print(f"\n[EVENT] Target reached at step {self.steps_taken}!")
            done = True
        
        # Check if battery depleted or max steps reached
        if self.battery_percentage <= 0 or self.steps_taken >= self.max_steps:
            if self.battery_percentage <= 0:
                print(f"\n[EVENT] Battery depleted at step {self.steps_taken}!")
            else:
                print(f"\n[EVENT] Maximum steps reached ({self.steps_taken}/{self.max_steps})!")
            done = True
        
        # Update previous distance
        self.prev_distance = self.current_distance
        
        # Set episode complete flag for the callback
        if done:
            self.episode_complete = True
            print(f"\n[Episode {self.episode_num}] Completed with reward {self.episode_reward:.2f} in {self.steps_taken} steps")
            print(f"[Episode {self.episode_num}] Final distance to target: {self.current_distance:.3f}")
            if self.collision_occurred:
                print(f"[Episode {self.episode_num}] Collisions occurred during this episode")
        
        # Return observation: [dx, dy, dz, battery_percentage]
        obs = np.array([dx, dy, dz, self.battery_percentage], dtype=np.float32)
        return obs, step_reward, done, {}

    def render(self, mode="human"):
        # Rendering is not implemented; see the simulation window.
        pass

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("[DroneEnv] Environment closed.")


# ----- Training script -----
if __name__ == "__main__":
    # Check for GPU availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using device: {device}")
    
    if device.type == "cuda":
        print(f"[Training] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Training] Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"[Training] Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    try:
        # Create the environment
        env = DroneEnv()
        
        # Create the PPO model with optimized parameters for GPU/CPU utilization
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=0,  # Reduce verbosity for cleaner output with progress bar
            learning_rate=0.0003,
            n_steps=1024,  # Collect more steps before updating for better GPU utilization
            batch_size=64,  # Larger batch size for GPU efficiency
            n_epochs=10,    # More epochs per update for better learning
            gamma=0.99,
            ent_coef=0.01,  # Encourage exploration
            device=device   # Use GPU if available
        )

        # Create a callback with progress bar
        progress_callback = ProgressBarCallback()

        # Train the model for exactly 20,000 steps (100 episodes × 200 steps)
        print("[Training] Starting training for 100 episodes (20,000 steps)...")
        model.learn(total_timesteps=20000, callback=progress_callback)

        # Save the trained model.
        model.save("final_drone_model")
        print("[Training] Model trained and saved as 'final_drone_model.zip'")
        
        # Also save the best model if it's different from the final one
        print("[Training] Best model saved as 'best_drone_model.zip'")

    except Exception as e:
        print(f"[Error] An error occurred: {e}")
    finally:
        if 'env' in locals():
            env.close()
        
    print("[Training] Process completed successfully.")

import gym
import numpy as np
import airsim
import time
import math
import os
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import subprocess

# ----- Optional: Launch Blocks environment automatically -----
def launch_airsim(gui_enabled=True):
    """
    Launch AirSim Blocks environment with GUI enabled or disabled.
    
    Args:
        gui_enabled (bool): Set to True to enable GUI, False to disable GUI
    """
    # Update this path to your built Blocks.exe executable if necessary.
    executable_path = r"C:\Programming\Reinforcement Learning\Autonomous Drone\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe"
    
    # Command line arguments to enable/disable GUI
    args = []
    if not gui_enabled:
        args.append("-windowed")
        args.append("-NoVSync")
        args.append("-RenderOffScreen")
    
    process = subprocess.Popen([executable_path] + args)
    print(f"[Launcher] Launching AirSim Blocks environment with GUI {'enabled' if gui_enabled else 'disabled'}...")
    # Wait enough time for the simulation to load.
    time.sleep(15)
    print("[Launcher] Blocks environment should be ready.")
    return process

# ----- Custom Callback to log episode rewards and progress and stop at max episodes -----
class DroneTrainingCallback(BaseCallback):
    def __init__(self, max_episodes=100, verbose=0):
        super(DroneTrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.best_reward = -float('inf')
        self.episode_count = 0
        self.max_episodes = max_episodes
        
    def _on_step(self) -> bool:
        # Update episode reward
        self.current_episode_reward += self.locals['rewards'][0]
        
        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            
            # Check if this is the best episode so far
            if self.current_episode_reward > self.best_reward:
                self.best_reward = self.current_episode_reward
                print(f"\n[Callback] New best episode reward: {self.best_reward:.2f}")
            
            # Reset for next episode
            self.current_episode_reward = 0
            
            # Check if we've reached the maximum number of episodes
            if self.episode_count >= self.max_episodes:
                print(f"\n[Callback] Reached {self.max_episodes} episodes. Stopping training.")
                return False  # Return False to stop training
            
        return True  # Continue training

# ----- Define the custom Gym environment -----
class DroneEnv(gym.Env):
    """A custom Gym environment for training an RL agent to control a drone in AirSim."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_episodes=100, max_steps_per_episode=200):
        super(DroneEnv, self).__init__()

        # Define the action space: 4 continuous actions [vx, vy, vz, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-5, -5, -2, -45], dtype=np.float32),
            high=np.array([5, 5, 2, 45], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: relative position of the drone to target: [dx, dy, dz]
        self.observation_space = spaces.Box(
            low=-100, 
            high=100, 
            shape=(3,), 
            dtype=np.float32
        )

        # Connect to AirSim
        self.client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
        self.client.confirmConnection()
        print("[DroneEnv] Connected to AirSim.")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Fixed start and target points (NED coordinates)
        # Convert from world coordinates to NED coordinates
        # In NED: x is North (+), y is East (+), z is Down (+)
        self.start_point = airsim.Vector3r(0.000, 0.000, -0.6777)  # Starting position
        self.target_point = airsim.Vector3r(-40.972, 0.878, -0.676)  # Target position
        
        # Target threshold (0.5 on each axis)
        self.target_threshold = 0.5
        
        # Duration for each action command
        self.duration = 0.5
        
        # Training parameters
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.episode_num = 0
        self.total_episodes = max_episodes
        
        # Battery simulation
        self.battery_percentage = 100
        self.battery_drain_rate = 0.5  # 0.5% per step (1% per 2 actions)
        
        # Reward tracking
        self.prev_distance = None
        self.episode_reward = 0
        self.collision_occurred = False
        
        # Current action tracking
        self.current_action = [0, 0, 0, 0]
        
        # Add a marker at the target position
        self.add_target_marker()

    def add_target_marker(self):
        """Add a visible marker at the target position."""
        # Create a large red box at the target position
        marker_size = 1.0  # 1 meter cube
        marker_pose = airsim.Pose(
            airsim.Vector3r(self.target_point.x_val, self.target_point.y_val, self.target_point.z_val),
            airsim.Quaternionr(0, 0, 0, 1)
        )
        
        # Try to create or update the marker
        try:
            self.client.simCreateVehicle("TargetMarker", "SimpleFlight", marker_pose, marker_size)
            print("[DroneEnv] Target marker created successfully.")
        except:
            print("[DroneEnv] Could not create target marker. Will continue without visual marker.")

    def reset(self):
        """Reset the environment at the beginning of a new episode."""
        self.episode_num += 1
        print(f"\n[Episode {self.episode_num}/{self.total_episodes}] Starting new episode")
        
        # Reset the drone to the starting position
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Teleport the drone to the starting position
        pose = airsim.Pose(
            airsim.Vector3r(self.start_point.x_val, self.start_point.y_val, self.start_point.z_val),
            airsim.Quaternionr(0, 0, 0, 1)
        )
        self.client.simSetVehiclePose(pose, True)
        
        # Hover in place to stabilize
        self.client.hoverAsync().join()
        time.sleep(1)  # Give it a moment to stabilize
        
        # Reset episode variables
        self.current_step = 0
        self.battery_percentage = 100
        self.episode_reward = 0
        self.collision_occurred = False
        self.current_action = [0, 0, 0, 0]
        
        # Get the initial observation
        obs = self._get_observation()
        self.prev_distance = self._get_distance_to_target()
        
        # Print initial status
        position = self._get_position()
        print(f"EP:{self.episode_num:03d} | STEP:{self.current_step:03d} | POS:({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}) | "
              f"DIST:{self.prev_distance:.2f} | BAT:{self.battery_percentage:.1f}% | "
              f"ACT:[{0:.2f},{0:.2f},{0:.2f},{0:.2f}] | REW:0 | EPREW:0")
        
        return obs

    def step(self, action):
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Store the current action for display
        self.current_action = action.tolist()
        
        # Execute the action
        vx, vy, vz, yaw_rate = action
        self.client.moveByVelocityAsync(float(vx), float(vy), float(vz), self.duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, float(yaw_rate)))
        time.sleep(self.duration)
        
        # Update battery
        self.battery_percentage -= self.battery_drain_rate
        self.battery_percentage = max(0, self.battery_percentage)
        
        # Get the new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward()
        self.episode_reward += reward
        
        # Check if episode is done
        done = self._is_done()
        
        # Create custom info dict for display
        display_info = {
            'position': self._get_position(),
            'distance': self._get_distance_to_target(),
            'battery': self.battery_percentage,
            'step': self.current_step,
            'episode': self.episode_num,
            'reward': reward,
            'episode_reward': self.episode_reward,
            'collision': self.collision_occurred,
            'action': self.current_action
        }
        
        # Print status update
        self._print_status(display_info, done)
        
        # Create proper info dict for stable_baselines3
        # This is the key fix: stable_baselines3 expects 'episode' to be a dict with episode stats
        info = {}
        
        # Only add episode info when episode is done
        if done:
            # Format episode info as expected by stable_baselines3
            episode_info = {
                'r': float(self.episode_reward),  # episode reward
                'l': int(self.current_step),      # episode length
                't': time.time()                  # episode end time
            }
            info['episode'] = episode_info
        
        return obs, reward, done, info

    def _get_observation(self):
        """Get the current observation (relative position to target)."""
        drone_state = self.client.getMultirotorState()
        drone_position = drone_state.kinematics_estimated.position
        
        # Calculate relative position to target (dx, dy, dz)
        dx = self.target_point.x_val - drone_position.x_val
        dy = self.target_point.y_val - drone_position.y_val
        dz = self.target_point.z_val - drone_position.z_val
        
        return np.array([dx, dy, dz], dtype=np.float32)

    def _get_position(self):
        """Get the current drone position."""
        drone_state = self.client.getMultirotorState()
        position = drone_state.kinematics_estimated.position
        return (position.x_val, position.y_val, position.z_val)

    def _get_distance_to_target(self):
        """Calculate the Euclidean distance to the target."""
        drone_state = self.client.getMultirotorState()
        drone_position = drone_state.kinematics_estimated.position
        
        dx = self.target_point.x_val - drone_position.x_val
        dy = self.target_point.y_val - drone_position.y_val
        dz = self.target_point.z_val - drone_position.z_val
        
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _compute_reward(self):
        """Compute the reward based on the current state."""
        # Check for collision
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided and not self.collision_occurred:
            self.collision_occurred = True
            print(f"\n[EVENT] Episode {self.episode_num}, Step {self.current_step}: Collision occurred!")
            return -20  # Penalty for collision
        
        # Calculate distance-based reward
        current_distance = self._get_distance_to_target()
        
        if self.prev_distance is not None:
            # If distance decreased, positive reward
            if current_distance < self.prev_distance:
                reward = 1
            # If distance increased, negative reward
            else:
                reward = -1
        else:
            reward = 0
        
        self.prev_distance = current_distance
        return reward

    def _is_done(self):
        """Check if the episode is done."""
        # Check if target reached
        current_distance = self._get_distance_to_target()
        target_reached = self._is_target_reached()
        
        if target_reached:
            print(f"\n[EVENT] Episode {self.episode_num}: Target reached in {self.current_step} steps!")
            return True
        
        # Check if battery depleted (max steps reached)
        if self.current_step >= self.max_steps_per_episode:
            print(f"\n[EVENT] Episode {self.episode_num}: Max steps reached, battery depleted!")
            return True
        
        return False

    def _is_target_reached(self):
        """Check if the drone has reached the target within the threshold."""
        drone_state = self.client.getMultirotorState()
        drone_position = drone_state.kinematics_estimated.position
        
        # Check if within threshold on each axis
        x_reached = abs(self.target_point.x_val - drone_position.x_val) <= self.target_threshold
        y_reached = abs(self.target_point.y_val - drone_position.y_val) <= self.target_threshold
        z_reached = abs(self.target_point.z_val - drone_position.z_val) <= self.target_threshold
        
        return x_reached and y_reached and z_reached

    def _print_status(self, info, done):
        """Print the current status of the drone in a compact, readable format."""
        position = info['position']
        action = info['action']
        
        # Format: EP:001 | STEP:005 | POS:(x,y,z) | DIST:42.5 | BAT:97.5% | ACT:[vx,vy,vz,yaw] | REW:1 | EPREW:5
        status_line = (
            f"EP:{info['episode']:03d} | "
            f"STEP:{info['step']:03d} | "
            f"POS:({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}) | "
            f"DIST:{info['distance']:.2f} | "
            f"BAT:{info['battery']:.1f}% | "
            f"ACT:[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f},{action[3]:.2f}] | "
            f"REW:{info['reward']} | "
            f"EPREW:{info['episode_reward']}"
        )
        
        # Print on same line with carriage return, unless episode is done
        if not done:
            print(status_line, end='\r')
        else:
            print(status_line)  # Print final status on a new line

    def close(self):
        """Clean up resources."""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("[DroneEnv] Environment closed.")


def train_drone(gui_enabled=True, max_episodes=100, max_steps_per_episode=200, save_path="./drone_model"):
    """
    Train the drone using PPO algorithm.
    
    Args:
        gui_enabled (bool): Whether to enable GUI during training
        max_episodes (int): Maximum number of episodes to train for
        max_steps_per_episode (int): Maximum number of steps per episode
        save_path (str): Path to save the trained model
    """
    # Launch AirSim if needed
    process = launch_airsim(gui_enabled=gui_enabled)
    
    try:
        # Create the environment with specified episode and step limits
        env = DroneEnv(max_episodes=max_episodes, max_steps_per_episode=max_steps_per_episode)
        
        # Create the callback with episode limit
        callback = DroneTrainingCallback(max_episodes=max_episodes)
        
        # Calculate a safe upper bound for total timesteps
        # This is just a large enough number to ensure we don't stop prematurely
        # The callback will stop training once max_episodes is reached
        total_timesteps = max_episodes * max_steps_per_episode * 2
        
        # Create and train the agent
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=min(2048, max_steps_per_episode * 10),  # Ensure n_steps is reasonable
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./drone_tensorboard/"
        )
        
        print(f"[Trainer] Starting training for up to {max_episodes} episodes with {max_steps_per_episode} steps per episode...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Save the model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"[Trainer] Model saved to {save_path}")
        
        return model, env
        
    except Exception as e:
        print(f"[Trainer] Error during training: {e}")
        raise
    finally:
        # Clean up
        if 'env' in locals():
            env.close()
        
        # Terminate AirSim process if we launched it
        if process:
            process.terminate()
            print("[Launcher] AirSim process terminated.")


if __name__ == "__main__":
    # Set to False to disable GUI during training (faster training)
    # Set to True to enable GUI (for visualization)
    GUI_ENABLED = True
    
    # Training parameters - modify these as needed
    MAX_EPISODES = 100
    MAX_STEPS_PER_EPISODE = 200
    
    # Train the drone with explicit episode and step limits
    model, env = train_drone(
        gui_enabled=GUI_ENABLED,
        max_episodes=MAX_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        save_path="./drone_model"
    )
    
    print("[Main] Training completed successfully!")

import gym
import numpy as np
import airsim
import time
import math
import os
import subprocess
from stable_baselines3 import PPO

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

# ----- Define the custom Gym environment -----
class DroneEnv(gym.Env):
    """A custom Gym environment for training an RL agent to control a drone in AirSim."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_steps_per_episode=200):
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
        print("\n[Test] Starting test episode")
        
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
        print(f"STEP:{self.current_step:03d} | POS:({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}) | "
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
            'reward': reward,
            'episode_reward': self.episode_reward,
            'collision': self.collision_occurred,
            'action': self.current_action
        }
        
        # Print status update
        self._print_status(display_info, done)
        
        return obs, reward, done, {}

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
            print(f"\n[EVENT] Step {self.current_step}: Collision occurred!")
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
            print(f"\n[EVENT] Target reached in {self.current_step} steps!")
            return True
        
        # Check if battery depleted (max steps reached)
        if self.current_step >= self.max_steps_per_episode:
            print(f"\n[EVENT] Max steps reached, battery depleted!")
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
        
        # Format: STEP:005 | POS:(x,y,z) | DIST:42.5 | BAT:97.5% | ACT:[vx,vy,vz,yaw] | REW:1 | EPREW:5
        status_line = (
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


def test_drone(model_path="./drone_model.zip", gui_enabled=True, max_steps_per_episode=200, num_test_episodes=3):
    """
    Test the trained drone model.
    
    Args:
        model_path (str): Path to the trained model file
        gui_enabled (bool): Whether to enable GUI during testing
        max_steps_per_episode (int): Maximum number of steps per episode
        num_test_episodes (int): Number of test episodes to run
    """
    # Launch AirSim if needed
    process = launch_airsim(gui_enabled=gui_enabled)
    
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"[Tester] Error: Model file '{model_path}' not found!")
            return
        
        print(f"[Tester] Loading model from {model_path}")
        
        # Create the environment with specified step limit
        env = DroneEnv(max_steps_per_episode=max_steps_per_episode)
        
        # Load the trained model
        model = PPO.load(model_path)
        print("[Tester] Model loaded successfully!")
        
        # Run test episodes
        print(f"[Tester] Running {num_test_episodes} test episodes...")
        
        episode_results = []
        
        for episode in range(num_test_episodes):
            print(f"\n[Tester] Test Episode {episode+1}/{num_test_episodes}")
            
            # Reset the environment
            obs = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            # Run one episode
            while not done:
                # Get the model's action
                action, _ = model.predict(obs, deterministic=True)
                
                # Execute the action
                obs, reward, done, _ = env.step(action)
                
                episode_reward += reward
                steps += 1
            
            # Record episode results
            target_reached = env._is_target_reached()
            result = {
                "episode": episode + 1,
                "steps": steps,
                "reward": episode_reward,
                "target_reached": target_reached,
                "collision": env.collision_occurred
            }
            episode_results.append(result)
            
            # Short pause between episodes
            time.sleep(2)
        
        # Print summary
        print("\n[Tester] Test Results Summary:")
        print("-" * 50)
        print(f"{'Episode':<10} {'Steps':<10} {'Reward':<10} {'Target':<10} {'Collision':<10}")
        print("-" * 50)
        
        targets_reached = 0
        collisions = 0
        total_steps = 0
        total_reward = 0
        
        for result in episode_results:
            print(f"{result['episode']:<10} {result['steps']:<10} {result['reward']:<10.1f} {'Yes' if result['target_reached'] else 'No':<10} {'Yes' if result['collision'] else 'No':<10}")
            
            if result['target_reached']:
                targets_reached += 1
            if result['collision']:
                collisions += 1
            total_steps += result['steps']
            total_reward += result['reward']
        
        print("-" * 50)
        print(f"Success Rate: {targets_reached/num_test_episodes*100:.1f}%")
        print(f"Collision Rate: {collisions/num_test_episodes*100:.1f}%")
        print(f"Average Steps: {total_steps/num_test_episodes:.1f}")
        print(f"Average Reward: {total_reward/num_test_episodes:.1f}")
        
    except Exception as e:
        print(f"[Tester] Error during testing: {e}")
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
    # Import necessary modules in the main scope
    from gym import spaces
    
    # Set to True to enable GUI during testing (visualization)
    GUI_ENABLED = True
    
    # Testing parameters
    MAX_STEPS_PER_EPISODE = 200  # Same as in training
    NUM_TEST_EPISODES = 3        # Number of test episodes to run
    
    # Path to the trained model
    MODEL_PATH = "./drone_model.zip"
    
    # Test the trained drone model
    test_drone(
        model_path=MODEL_PATH,
        gui_enabled=GUI_ENABLED,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        num_test_episodes=NUM_TEST_EPISODES
    )
    
    print("[Main] Testing completed successfully!")
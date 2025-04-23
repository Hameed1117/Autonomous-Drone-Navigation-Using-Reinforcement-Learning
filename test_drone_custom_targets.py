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
    """A custom Gym environment for testing a drone in AirSim with different target locations."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, target_point=None, max_steps_per_episode=300, target_marker_size=3.0):
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

        # Fixed start point (NED coordinates)
        self.start_point = airsim.Vector3r(0.000, 0.000, -0.6777)  # Starting position
        
        # Target point (can be customized)
        if target_point is None:
            # Default target from training
            self.target_point = airsim.Vector3r(-40.972, 0.878, -0.676)
        else:
            self.target_point = airsim.Vector3r(target_point[0], target_point[1], target_point[2])
        
        print(f"[DroneEnv] Target set to: ({self.target_point.x_val}, {self.target_point.y_val}, {self.target_point.z_val})")
        
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
        
        # Target marker size (larger value = more visible)
        self.target_marker_size = target_marker_size
        
        # Add visual indicators in the environment
        self.add_environment_markers()

    def add_environment_markers(self):
        """Add visual indicators in the environment"""
        try:
            # 1. Add a large marker at the target position
            self.add_target_marker()
            
            # 2. Add a starting position marker (green)
            self.add_start_marker()
            
            # 3. Create a line of small markers from start to target to show direction
            self.add_path_markers()
            
        except Exception as e:
            print(f"[DroneEnv] Warning: Could not create all markers: {e}")
            print("[DroneEnv] Will continue without some visual indicators.")

    def add_target_marker(self):
        """Add a visible marker at the target position."""
        try:
            # First try to remove any existing marker
            try:
                self.client.simDestroyObject("TargetMarker")
                time.sleep(0.1)
            except:
                pass
            
            # Create a large red box at the target position
            marker_pose = airsim.Pose(
                airsim.Vector3r(self.target_point.x_val, self.target_point.y_val, self.target_point.z_val),
                airsim.Quaternionr(0, 0, 0, 1)
            )
            
            # Try different methods to create a visible marker
            try:
                # Method 1: Create a vehicle as a marker (more visible)
                self.client.simAddVehicle("TargetMarker", "ComputerVision", marker_pose)
                print("[DroneEnv] Target marker created successfully (vehicle method).")
            except:
                try:
                    # Method 2: Use objects API
                    self.client.simSpawnObject("TargetMarker", "cube", marker_pose, 
                                             airsim.Vector3r(self.target_marker_size, self.target_marker_size, self.target_marker_size),
                                             is_static=True)
                    print("[DroneEnv] Target marker created successfully (object method).")
                except:
                    # Method 3: Old API for backwards compatibility
                    self.client.simCreateVehicle("TargetMarker", "SimpleFlight", marker_pose, self.target_marker_size)
                    print("[DroneEnv] Target marker created successfully (legacy method).")
            
            # Add floating text above target
            target_text_pose = airsim.Pose(
                airsim.Vector3r(self.target_point.x_val, self.target_point.y_val, self.target_point.z_val - 2.0),
                airsim.Quaternionr(0, 0, 0, 1)
            )
            try:
                self.client.simAddText3D("TARGET", target_text_pose, 
                                       text_size_cm=200, text_color_rgba=[1, 0, 0, 1])
                print("[DroneEnv] Target text label added.")
            except:
                print("[DroneEnv] Could not add 3D text label.")
            
        except Exception as e:
            print(f"[DroneEnv] Could not create target marker: {e}")
            print("[DroneEnv] Will continue without visual marker.")
    
    def add_start_marker(self):
        """Add a marker at the starting position."""
        try:
            # Create a green box at the starting position
            start_pose = airsim.Pose(
                airsim.Vector3r(self.start_point.x_val, self.start_point.y_val, self.start_point.z_val),
                airsim.Quaternionr(0, 0, 0, 1)
            )
            
            try:
                self.client.simSpawnObject("StartMarker", "cube", start_pose, 
                                         airsim.Vector3r(1.0, 1.0, 1.0),
                                         is_static=True)
                print("[DroneEnv] Start marker created successfully.")
            except:
                print("[DroneEnv] Could not create start marker.")
        except:
            print("[DroneEnv] Could not create start marker.")
    
    def add_path_markers(self):
        """Add a series of small markers to indicate path from start to target."""
        try:
            # Calculate vector from start to target
            dx = self.target_point.x_val - self.start_point.x_val
            dy = self.target_point.y_val - self.start_point.y_val
            dz = self.target_point.z_val - self.start_point.z_val
            
            # Number of markers to place
            num_markers = 5
            
            # Create markers along the path
            for i in range(1, num_markers):
                # Calculate position (evenly spaced)
                factor = i / num_markers
                pos_x = self.start_point.x_val + dx * factor
                pos_y = self.start_point.y_val + dy * factor
                pos_z = self.start_point.z_val + dz * factor
                
                marker_pose = airsim.Pose(
                    airsim.Vector3r(pos_x, pos_y, pos_z),
                    airsim.Quaternionr(0, 0, 0, 1)
                )
                
                try:
                    # Create a small sphere
                    marker_name = f"PathMarker_{i}"
                    self.client.simSpawnObject(marker_name, "sphere", marker_pose, 
                                             airsim.Vector3r(0.5, 0.5, 0.5),
                                             is_static=True)
                except:
                    # If spawn fails, try alternative method
                    pass
            
            print("[DroneEnv] Path markers created successfully.")
        except:
            print("[DroneEnv] Could not create path markers.")

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
        
        # Refresh markers to ensure they're visible
        self.add_environment_markers()
        
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


def test_drone_custom_targets(model_path="./drone_model.zip", gui_enabled=True, max_steps_per_episode=300):
    """
    Test the trained drone model with specific target locations.
    
    Args:
        model_path (str): Path to the trained model file
        gui_enabled (bool): Whether to enable GUI during testing
        max_steps_per_episode (int): Maximum number of steps per episode
    """
    # Launch AirSim if needed
    process = launch_airsim(gui_enabled=gui_enabled)
    
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"[Tester] Error: Model file '{model_path}' not found!")
            return
        
        print(f"[Tester] Loading model from {model_path}")
        
        # Load the trained model
        model = PPO.load(model_path)
        print("[Tester] Model loaded successfully!")
        
        # Define the custom target locations to test (x, y, z in NED coordinates)
        target_locations = [
            (-40.972, 0.878, -0.676),        # Original training target
            (-35.480, -35.871, -19.422),     # Custom target 1
            (-44.536, 26.959, 0.668),        # Custom target 2
            (32.974, 32.932, -8.931)         # Custom target 3
        ]
        
        target_names = [
            "Original Training Target",
            "Custom Target 1",
            "Custom Target 2", 
            "Custom Target 3"
        ]
        
        results = []
        
        # Test each target location
        for i, target in enumerate(target_locations):
            print(f"\n[Tester] Testing target {i+1}/{len(target_locations)}: {target_names[i]}")
            print(f"[Tester] Coordinates: ({target[0]}, {target[1]}, {target[2]})")
            
            # Create environment with this target and larger marker
            env = DroneEnv(target_point=target, max_steps_per_episode=max_steps_per_episode, target_marker_size=5.0)
            
            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            start_time = time.time()
            
            # Run one episode
            while not done:
                # Get the model's action
                action, _ = model.predict(obs, deterministic=True)
                
                # Execute the action
                obs, reward, done, _ = env.step(action)
                
                episode_reward += reward
                steps += 1
            
            # Record results
            end_time = time.time()
            duration = end_time - start_time
            target_reached = env._is_target_reached()
            distance = env._get_distance_to_target()
            result = {
                "target_num": i + 1,
                "target_name": target_names[i],
                "target": target,
                "steps": steps,
                "duration": duration,
                "reward": episode_reward,
                "reached": target_reached,
                "final_distance": distance,
                "collision": env.collision_occurred
            }
            results.append(result)
            
            # Close environment
            env.close()
            
            # Short pause between episodes
            time.sleep(2)
        
        # Print summary
        print("\n[Tester] Test Results Summary:")
        print("-" * 100)
        print(f"{'#':<3} {'Target':<25} {'Steps':<8} {'Time':<8} {'Reached':<8} {'Final Dist':<12} {'Collision':<9}")
        print("-" * 100)
        
        targets_reached = 0
        
        for r in results:
            target_str = f"{r['target_name']}"
            print(f"{r['target_num']:<3} {target_str:<25} {r['steps']:<8} {r['duration']:.1f}s {('Yes' if r['reached'] else 'No'):<8} {r['final_distance']:.2f} m{' ':<8} {'Yes' if r['collision'] else 'No':<9}")
            
            if r['reached']:
                targets_reached += 1
        
        print("-" * 100)
        print(f"Success Rate: {targets_reached/len(target_locations)*100:.1f}% ({targets_reached}/{len(target_locations)})")
        
    except Exception as e:
        print(f"[Tester] Error during testing: {e}")
        raise
    finally:
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
    MAX_STEPS_PER_EPISODE = 300  # Increased from 200 for longer distance targets
    
    # Path to the trained model
    MODEL_PATH = "./drone_model.zip"
    
    # Test the trained drone model with custom targets
    test_drone_custom_targets(
        model_path=MODEL_PATH,
        gui_enabled=GUI_ENABLED,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE
    )
    
    print("[Main] Testing completed successfully!")
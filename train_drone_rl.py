import gym
import numpy as np
import airsim
import time
import math
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import subprocess


# ----- Optional: Launch Blocks environment automatically -----
def launch_airsim():
    # Update this path to your built Blocks.exe executable if necessary.
    executable_path = r"C:\Programming\Reinforcement Learning\Autonomous Drone\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe"
    process = subprocess.Popen(executable_path)
    print("[Launcher] Launching AirSim Blocks environment...")
    # Wait enough time for the simulation to load.
    time.sleep(15)
    print("[Launcher] Blocks environment should be ready.")
    return process


# ----- Custom Callback to log step counts -----
class StepLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(StepLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Print step count every 1000 steps.
        if self.n_calls % 10 == 0:
            print(f"[Callback] Total timesteps: {self.num_timesteps}")
        return True


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

        # Observation: relative position of the drone to target: [dx, dy, dz]
        self.observation_space = spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)

        # Connect to AirSim.
        self.client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
        self.client.confirmConnection()
        print("[DroneEnv] Connected to AirSim.")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Fixed start and target points (NED coordinates).
        self.start_point = airsim.Vector3r(30, 10, -10)  # Starting position.
        self.target_point = airsim.Vector3r(90, 20, -10)  # Target position.
        self.target_threshold = 2.0  # Reaching within 5 meters is considered success.

        self.duration = 0.5  # Duration for each action command.
        self.prev_distance = None

    def reset(self):
        print("[DroneEnv] Resetting environment...")
        self.client.reset()
        time.sleep(1)

        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Take off.
        self.client.takeoffAsync().join()
        time.sleep(2)

        # Move the drone to the start point.
        print(
            f"[DroneEnv] Moving to start point: ({self.start_point.x_val}, {self.start_point.y_val}, {self.start_point.z_val})")
        self.client.moveToPositionAsync(self.start_point.x_val, self.start_point.y_val, self.start_point.z_val,
                                        5).join()
        time.sleep(2)

        # Plot the start and target markers in the environment.
        # The start point will be a red marker; the target point a green marker.
        # Note: simPlotPoints draws points that persist for a set duration.
        self.client.simPlotPoints([self.start_point], color_rgba=[1, 0, 0, 1], size=10.0, duration=60,
                                  is_persistent=True)
        self.client.simPlotPoints([self.target_point], color_rgba=[0, 1, 0, 1], size=10.0, duration=60,
                                  is_persistent=True)

        # Get initial observation.
        pose = self.client.simGetVehiclePose().position
        obs = np.array([
            self.target_point.x_val - pose.x_val,
            self.target_point.y_val - pose.y_val,
            self.target_point.z_val - pose.z_val
        ], dtype=np.float32)
        self.prev_distance = np.linalg.norm(obs)
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
        time.sleep(0.5)

        # Get new observation.
        pose = self.client.simGetVehiclePose().position
        obs = np.array([
            self.target_point.x_val - pose.x_val,
            self.target_point.y_val - pose.y_val,
            self.target_point.z_val - pose.z_val
        ], dtype=np.float32)
        current_distance = np.linalg.norm(obs)

        # Reward: reduction in distance with a small time penalty.
        reward = self.prev_distance - current_distance - 0.01

        done = False
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            reward -= 100
            done = True
            print("[DroneEnv] Collision occurred!")
        if current_distance < self.target_threshold:
            reward += 100
            done = True
            print("[DroneEnv] Target reached!")
        if pose.z_val > -1:  # Drone is too close to ground.
            reward -= 100
            done = True
            print("[DroneEnv] Drone has landed prematurely!")

        self.prev_distance = current_distance
        info = {}
        return obs, reward, done, info

    def render(self, mode="human"):
        # Rendering is not implemented; see the simulation window.
        pass

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)


# ----- Training script -----
if __name__ == "__main__":
    # Optionally, launch the Blocks environment.
    sim_process = launch_airsim()

    try:
        env = DroneEnv()

        # Create the PPO model using an MLP policy.
        model = PPO("MlpPolicy", env, verbose=1)

        # Create a callback for logging steps.
        step_callback = StepLoggingCallback()

        # Train the model.
        total_timesteps = 100  # Adjust training duration as needed.
        model.learn(total_timesteps=total_timesteps, callback=step_callback)

        # Save the trained model.
        model.save("drone_ppo_model")
        print("[Training] Model trained and saved as 'drone_ppo_model.zip'")

        # Optionally, later you can load the model with:
        # model = PPO.load("drone_ppo_model", env=env)

    finally:
        env.close()
        print("[Launcher] Terminating simulation...")
        sim_process.terminate()

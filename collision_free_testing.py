# Filename: testing.py

import gym
import numpy as np
import airsim
import time
import math
import os
import subprocess
from stable_baselines3 import PPO
from gym import spaces

# ----- Launch AirSim Blocks environment -----
def launch_airsim(gui_enabled=True):
    executable_path = r"C:\Programming\Reinforcement Learning\Autonomous Drone\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe"
    args = []
    if not gui_enabled:
        args.append("-windowed")
        args.append("-NoVSync")
        args.append("-RenderOffScreen")
    process = subprocess.Popen([executable_path] + args)
    print(f"[Launcher] Launching AirSim Blocks environment with GUI {'enabled' if gui_enabled else 'disabled'}...")
    time.sleep(15)
    print("[Launcher] Blocks environment should be ready.")
    return process

# ----- Custom Drone Test Environment -----
class DroneTestEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, target_point, max_steps_per_episode=300):
        super(DroneTestEnv, self).__init__()

        self.action_space = spaces.Box(
            low=np.array([-5, -5, -2, -45], dtype=np.float32),
            high=np.array([5, 5, 2, 45], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-100,
            high=100,
            shape=(3,),
            dtype=np.float32
        )

        self.client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
        self.client.confirmConnection()
        print("[DroneTestEnv] Connected to AirSim.")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.start_point = airsim.Vector3r(0.000, 0.000, -0.6777)
        self.target_point = airsim.Vector3r(target_point[0], target_point[1], target_point[2])

        self.target_threshold = 0.5
        self.duration = 0.5
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.battery_percentage = 100
        self.battery_drain_rate = 0.5
        self.prev_distance = None
        self.episode_reward = 0
        self.collision_occurred = False
        self.initial_collision_ignore_steps = 2
        self.steps_since_reset = 0
        self.current_action = [0, 0, 0, 0]

    def reset(self):
        print("\n[Tester] Starting test episode")

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        pose = airsim.Pose(self.start_point, airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose, True)
        self.client.hoverAsync().join()
        time.sleep(1)

        self.current_step = 0
        self.steps_since_reset = 0
        self.battery_percentage = 100
        self.episode_reward = 0
        self.collision_occurred = False
        self.current_action = [0, 0, 0, 0]

        obs = self._get_observation()
        self.prev_distance = self._get_distance_to_target()

        return obs

    def step(self, action):
        self.current_step += 1
        self.steps_since_reset += 1
        self.current_action = action.tolist()

        try:
            self.client.moveByVelocityAsync(
                float(action[0]), float(action[1]), float(action[2]), self.duration,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(True, float(action[3]))
            )
        except:
            pass

        time.sleep(self.duration)

        self.battery_percentage -= self.battery_drain_rate
        self.battery_percentage = max(0, self.battery_percentage)

        obs = self._get_observation()
        reward = self._compute_reward()
        self.episode_reward += reward
        done = self._is_done()

        self._print_status(reward, done)

        return obs, reward, done, {}

    def _print_status(self, reward, done):
        position = self._get_position()
        distance = self._get_distance_to_target()
        status_line = (
            f"STEP:{self.current_step:03d} | POS:({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}) | "
            f"DIST:{distance:.2f}m | BAT:{self.battery_percentage:.1f}% | "
            f"ACT:[{self.current_action[0]:.2f},{self.current_action[1]:.2f},{self.current_action[2]:.2f},{self.current_action[3]:.2f}] | "
            f"REW:{reward} | EPREW:{self.episode_reward}"
        )
        if not done:
            print(status_line, end='\r')
        else:
            print(status_line)
            summary = (
                f"[SUMMARY] Steps: {self.current_step} | Final Dist: {distance:.2f}m | "
                f"Reached: {'YES' if self._is_target_reached() else 'NO'}"
            )
            print(summary)

    def _get_observation(self):
        drone_pos = self.client.getMultirotorState().kinematics_estimated.position
        dx = self.target_point.x_val - drone_pos.x_val
        dy = self.target_point.y_val - drone_pos.y_val
        dz = self.target_point.z_val - drone_pos.z_val
        return np.array([dx, dy, dz], dtype=np.float32)

    def _get_position(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return (pos.x_val, pos.y_val, pos.z_val)

    def _get_distance_to_target(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        dx = self.target_point.x_val - pos.x_val
        dy = self.target_point.y_val - pos.y_val
        dz = self.target_point.z_val - pos.z_val
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _compute_reward(self):
        try:
            collision = self.client.simGetCollisionInfo()
            if collision.has_collided and not self.collision_occurred:
                self.collision_occurred = True
                print(f"\n[EVENT] Step {self.current_step}: Collision occurred!")
                return -20
        except:
            return 0

        dist_now = self._get_distance_to_target()
        if self.prev_distance is not None:
            reward = 1 if dist_now < self.prev_distance else -1
        else:
            reward = 0
        self.prev_distance = dist_now
        return reward

    def _is_done(self):
        if self._is_target_reached():
            print(f"\n[EVENT] Target reached at step {self.current_step}!")
            return True
        if self.current_step >= self.max_steps_per_episode:
            print(f"\n[EVENT] Max steps reached. Battery depleted.")
            return True
        return False

    def _is_target_reached(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        x_reached = abs(self.target_point.x_val - pos.x_val) <= self.target_threshold
        y_reached = abs(self.target_point.y_val - pos.y_val) <= self.target_threshold
        z_reached = abs(self.target_point.z_val - pos.z_val) <= self.target_threshold
        return x_reached and y_reached and z_reached

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("[DroneTestEnv] Environment closed.")

# ---- Test Function ----
def test_drone(model_path, gui_enabled=True, max_steps_per_episode=300):
    process = launch_airsim(gui_enabled)
    try:
        model = PPO.load(model_path)
        print(f"[Tester] Loaded model from {model_path}")

        # Target locations
        target_points = [
            (-40.972, 0.878, -0.676),
            (-35.480, -35.871, -19.422),
            (-44.536, 26.959, 0.668),
            (32.974, 32.932, -8.931)
        ]

        for idx, target in enumerate(target_points):
            print(f"\n[Tester] Testing Target {idx+1}: {target}")
            env = DroneTestEnv(target_point=target, max_steps_per_episode=max_steps_per_episode)
            obs = env.reset()

            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)

            env.close()
            time.sleep(2)

    finally:
        if process:
            process.terminate()
            print("[Launcher] AirSim process terminated.")

# ---- Main ----
if __name__ == "__main__":
    GUI_ENABLED = True
    MAX_STEPS = 300
    MODEL_PATH = r"C:\Programming\Reinforcement Learning\Autonomous Drone\collision_free_ppo_drone_20250429_044338.zip"

    test_drone(
        model_path=MODEL_PATH,
        gui_enabled=GUI_ENABLED,
        max_steps_per_episode=MAX_STEPS
    )

    print("[Main] Testing completed successfully!")

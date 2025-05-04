# Filename: collision_free_training.py

import gym
import numpy as np
import airsim
import time
import math
import os
import random
import datetime
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import subprocess

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

# ----- Custom Callback -----
class DroneTrainingCallback(BaseCallback):
    def __init__(self, max_episodes=100, verbose=0):
        super(DroneTrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.best_reward = -float('inf')
        self.episode_count = 0
        self.max_episodes = max_episodes

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            if self.current_episode_reward > self.best_reward:
                self.best_reward = self.current_episode_reward
                print(f"\n[Callback] New best episode reward: {self.best_reward:.2f}")
            self.current_episode_reward = 0
            if self.episode_count >= self.max_episodes:
                print(f"\n[Callback] Reached {self.max_episodes} episodes. Stopping training.")
                return False
        return True

# ----- Drone Environment -----
class DroneEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, max_episodes=100, max_steps_per_episode=200):
        super(DroneEnv, self).__init__()

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
        print("[DroneEnv] Connected to AirSim.")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.start_point = airsim.Vector3r(0.000, 0.000, -0.6777)
        self.target_locations = [
            airsim.Vector3r(-40.972, 0.878, -0.676),
            airsim.Vector3r(-35.480, -35.871, -19.422),
            airsim.Vector3r(-44.536, 26.959, 0.668),
            airsim.Vector3r(32.974, 32.932, -8.931)
        ]

        self.target_threshold = 0.5
        self.duration = 0.5
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.episode_num = 0
        self.total_episodes = max_episodes
        self.battery_percentage = 100
        self.battery_drain_rate = 0.5
        self.prev_distance = None
        self.episode_reward = 0
        self.collision_occurred = False
        self.initial_collision_ignore_steps = 2
        self.steps_since_reset = 0
        self.current_action = [0, 0, 0, 0]
        self.marker_error_printed = False

    def add_target_marker(self):
        try:
            points = [self.target_point]
            colors = [(1.0, 0.0, 0.0)]  # Red
            size = 30.0
            duration = 60.0
            is_persistent = True
            self.client.simPlotPoints(points, colors, size, duration, is_persistent)
            print("[DroneEnv] Target marker plotted successfully.")
        except Exception:
            if not self.marker_error_printed:
                print("Target marker implementation in work for future runs")
                self.marker_error_printed = True

    def reset(self):
        self.episode_num += 1
        print(f"\n[Episode {self.episode_num}/{self.total_episodes}] Starting new episode")

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.target_point = random.choice(self.target_locations)
        self.add_target_marker()

        pose = airsim.Pose(
            airsim.Vector3r(self.start_point.x_val, self.start_point.y_val, self.start_point.z_val),
            airsim.Quaternionr(0, 0, 0, 1)
        )
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

        noisy_action = action + np.random.normal(0, 0.2, size=4)
        vx, vy, vz, yaw_rate = noisy_action

        try:
            self.client.moveByVelocityAsync(
                float(vx), float(vy), float(vz), self.duration,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(True, float(yaw_rate))
            )
        except Exception as e:
            print(f"\n[ERROR] moveByVelocityAsync() failed: {e}")
            time.sleep(0.5)

        time.sleep(self.duration)

        self.battery_percentage -= self.battery_drain_rate
        self.battery_percentage = max(0, self.battery_percentage)

        obs = self._get_observation()
        reward = self._compute_reward()
        self.episode_reward += reward
        done = self._is_done()

        info = {}
        if done:
            episode_info = {
                'r': float(self.episode_reward),
                'l': int(self.current_step),
                't': time.time()
            }
            info['episode'] = episode_info

        self._print_status(reward, done)

        return obs, reward, done, info

    def _print_status(self, reward, done):
        position = self._get_position()
        action = self.current_action
        distance = self._get_distance_to_target()

        status_line = (
            f"EP:{self.episode_num:03d} | "
            f"STEP:{self.current_step:03d} | "
            f"ACT:[{action[0]:.2f},{action[1]:.2f},{action[2]:.2f},{action[3]:.2f}] | "
            f"BAT:{self.battery_percentage:.1f}% | "
            f"DIST:{distance:.2f}m | "
            f"REW:{reward} | "
            f"EPREW:{self.episode_reward}"
        )

        if not done:
            print(status_line, end='\r')
        else:
            print(status_line)
            target_reached = self._is_target_reached()
            summary_line = (
                f"[SUMMARY] Episode {self.episode_num} | "
                f"Episode Reward: {self.episode_reward:.2f} | "
                f"Final Distance to Target: {distance:.2f}m | "
                f"Target Reached: {'YES' if target_reached else 'NO'}"
            )
            print(summary_line)

    def _get_position(self):
        drone_state = self.client.getMultirotorState()
        position = drone_state.kinematics_estimated.position
        return (position.x_val, position.y_val, position.z_val)

    def _get_observation(self):
        drone_state = self.client.getMultirotorState()
        drone_position = drone_state.kinematics_estimated.position
        dx = self.target_point.x_val - drone_position.x_val
        dy = self.target_point.y_val - drone_position.y_val
        dz = self.target_point.z_val - drone_position.z_val
        return np.array([dx, dy, dz], dtype=np.float32)

    def _get_distance_to_target(self):
        drone_state = self.client.getMultirotorState()
        drone_position = drone_state.kinematics_estimated.position
        dx = self.target_point.x_val - drone_position.x_val
        dy = self.target_point.y_val - drone_position.y_val
        dz = self.target_point.z_val - drone_position.z_val
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _compute_reward(self):
        try:
            collision_info = self.client.simGetCollisionInfo()
        except Exception as e:
            print(f"\n[ERROR] simGetCollisionInfo() failed: {e}")
            return 0

        if collision_info.has_collided and not self.collision_occurred:
            if self.steps_since_reset > self.initial_collision_ignore_steps:
                self.collision_occurred = True
                print(f"\n[EVENT] Episode {self.episode_num}, Step {self.current_step}: Collision occurred!")
                self.current_action[3] += random.choice([-20, 20])
                return -20
            else:
                return 0

        current_distance = self._get_distance_to_target()

        if self.prev_distance is not None:
            if current_distance < self.prev_distance:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0

        self.prev_distance = current_distance
        return reward

    def _is_done(self):
        if self._is_target_reached():
            print(f"\n[EVENT] Episode {self.episode_num}: Target reached in {self.current_step} steps!")
            return True
        if self.current_step >= self.max_steps_per_episode:
            print(f"\n[EVENT] Episode {self.episode_num}: Max steps reached, battery depleted!")
            return True
        return False

    def _is_target_reached(self):
        drone_state = self.client.getMultirotorState()
        drone_position = drone_state.kinematics_estimated.position
        x_reached = abs(self.target_point.x_val - drone_position.x_val) <= self.target_threshold
        y_reached = abs(self.target_point.y_val - drone_position.y_val) <= self.target_threshold
        z_reached = abs(self.target_point.z_val - drone_position.z_val) <= self.target_threshold
        return x_reached and y_reached and z_reached

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("[DroneEnv] Environment closed.")

# ---- Train Function ----
def train_drone(gui_enabled=True, max_episodes=100, max_steps_per_episode=200, save_path="./collision_free_ppo_drone"):
    process = launch_airsim(gui_enabled=gui_enabled)
    model = None
    try:
        env = DroneEnv(max_episodes=max_episodes, max_steps_per_episode=max_steps_per_episode)
        callback = DroneTrainingCallback(max_episodes=max_episodes)
        total_timesteps = max_episodes * max_steps_per_episode * 2

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=min(2048, max_steps_per_episode * 10),
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./drone_tensorboard/"
        )

        print(f"[Trainer] Starting training for {max_episodes} episodes...")
        try:
            model.learn(total_timesteps=total_timesteps, callback=callback)
        except KeyboardInterrupt:
            print("\n[Trainer] Training manually interrupted. Saving model so far...")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"[Trainer] Model saved to {save_path}")

        return model, env

    finally:
        if 'env' in locals():
            env.close()
        if process:
            process.terminate()
            print("[Launcher] AirSim process terminated.")

# ---- Main ----
if __name__ == "__main__":
    GUI_ENABLED = True
    MAX_EPISODES = 148
    MAX_STEPS_PER_EPISODE = 200

    use_custom_name = False
    custom_name = "./collision_free_ppo_drone_1"

    if use_custom_name:
        model_save_path = custom_name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"./collision_free_ppo_drone_{timestamp}"

    model, env = train_drone(
        gui_enabled=GUI_ENABLED,
        max_episodes=MAX_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        save_path=model_save_path
    )

    print(f"[Main] Training completed successfully! Model saved at {model_save_path}")

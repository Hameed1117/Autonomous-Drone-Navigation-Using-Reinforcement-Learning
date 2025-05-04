import gym
import numpy as np
import airsim
import time
import math
import os
import random
import datetime
import matplotlib.pyplot as plt
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import subprocess

# --- Launch AirSim ---
def launch_airsim(gui_enabled=True):
    executable_path = r"C:\Programming\Reinforcement Learning\Autonomous Drone\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe"
    args = []
    if not gui_enabled:
        args += ["-windowed", "-NoVSync", "-RenderOffScreen"]
    process = subprocess.Popen([executable_path] + args)
    print(f"[Launcher] Launching AirSim Blocks environment with GUI {'enabled' if gui_enabled else 'disabled'}...")
    time.sleep(15)
    print("[Launcher] Blocks environment should be ready.")
    return process

# --- Training Callback ---
class DroneTrainingCallback(BaseCallback):
    def __init__(self, max_episodes=100, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.current_reward = 0

    def _on_step(self):
        self.current_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.locals['infos'][0].get("episode", {}).get("l", 0))
            self.episode_count += 1
            self.current_reward = 0
            if self.episode_count >= self.max_episodes:
                self.plot_training()
                return False
        return True

    def plot_training(self):
        episodes = list(range(1, len(self.episode_lengths) + 1))
        plt.figure(figsize=(10, 6))
        plt.bar(episodes, self.episode_lengths, color='green')
        plt.axhline(200, linestyle='--', color='red', label='Max Steps')
        plt.xlabel("Episode")
        plt.ylabel("Steps Taken")
        plt.title("Drone Navigation Steps per Episode")
        plt.legend()
        plt.tight_layout()
        plt.savefig("drone_training_performance.png")
        print("\n[Plot] Training plot saved as drone_training_performance.png")

# --- Drone Environment ---
class DroneEnv(gym.Env):
    def __init__(self, max_episodes=100, max_steps_per_episode=200):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([-5, -5, -2, -45], dtype=np.float32),
                                       high=np.array([5, 5, 2, 45], dtype=np.float32))
        self.observation_space = spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)

        self.client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
        self.client.confirmConnection()
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
        self.total_episodes = max_episodes
        self.episode_num = 0

    def reset(self):
        self.episode_num += 1
        print(f"\n[Episode {self.episode_num}/{self.total_episodes}] Starting new episode")

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.target_point = random.choice(self.target_locations)
        pose = airsim.Pose(self.start_point, airsim.to_quaternion(0, 0, 0))
        self.client.simSetVehiclePose(pose, True)
        self.client.hoverAsync().join()

        self.current_step = 0
        self.battery_percentage = 100
        self.episode_reward = 0
        self.prev_distance = self._get_distance_to_target()
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        vx, vy, vz, yaw_rate = action + np.random.normal(0, 0.2, size=4)

        # Face direction of movement
        yaw_angle = math.degrees(math.atan2(vy, vx))
        yaw_rate = yaw_angle

        # --- Lidar-based obstacle detection (ignoring ground) ---
        lidar_data = self.client.getLidarData("LidarSensor1")
        obstacle_ahead = False
        obstacle_high = False

        if len(lidar_data.point_cloud) >= 3:
            points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
            ground_threshold = -1.5
            valid_points = points[points[:, 2] > ground_threshold]

            if len(valid_points) > 0:
                distances = np.linalg.norm(valid_points, axis=1)
                forward_hits = valid_points[(valid_points[:, 0] > 0) & (distances < 4)]

                if len(forward_hits) > 10:
                    obstacle_ahead = True
                    max_height = np.max(forward_hits[:, 2])
                    if max_height > 1.5:
                        up_hits = valid_points[(valid_points[:, 2] > 1.5) & (valid_points[:, 0] > 0)]
                        if len(up_hits) > 10:
                            obstacle_high = True

        if obstacle_ahead:
            print("\n[Obstacle] Obstacle ahead detected.")
            if not obstacle_high:
                print("[Decision] Flying over small obstacle.")
                vz = 2.0
            else:
                print("[Decision] Tall obstacle detected. Turning 45°.")
                vx, vy = 0.0, 0.0
                yaw_rate += 45

        self.client.moveByVelocityAsync(float(vx), float(vy), float(vz), self.duration,
                                        airsim.DrivetrainType.MaxDegreeOfFreedom,
                                        airsim.YawMode(True, float(yaw_rate))).join()

        time.sleep(self.duration)
        self.battery_percentage -= 0.5
        obs = self._get_observation()

        # Continue movement even after collision
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            print(f"\n[EVENT] Collision detected at step {self.current_step}. Continuing...")

        reward = self._compute_reward()
        done = self._is_done()
        info = {'episode': {'r': self.episode_reward, 'l': self.current_step, 't': time.time()}}

        print(f"Step:{self.current_step:03} | Action:[{vx:.1f},{vy:.1f},{vz:.1f},{yaw_rate:.1f}] "
              f"| Dist:{self._get_distance_to_target():.2f} | Bat:{self.battery_percentage:.1f}% "
              f"| Rew:{reward:.1f}", end='\r')

        if done:
            print(f"\n[SUMMARY] Episode {self.episode_num} | Reward: {self.episode_reward:.2f} | "
                  f"Target Reached: {'YES' if self._is_target_reached() else 'NO'}")

        return obs, reward, done, info

    def _get_observation(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([
            self.target_point.x_val - pos.x_val,
            self.target_point.y_val - pos.y_val,
            self.target_point.z_val - pos.z_val
        ], dtype=np.float32)

    def _get_distance_to_target(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return math.sqrt(
            (self.target_point.x_val - pos.x_val)**2 +
            (self.target_point.y_val - pos.y_val)**2 +
            (self.target_point.z_val - pos.z_val)**2
        )

    def _compute_reward(self):
        dist = self._get_distance_to_target()
        reward = 1 if dist < self.prev_distance else -1
        self.prev_distance = dist
        self.episode_reward += reward
        return reward

    def _is_done(self):
        return self._is_target_reached() or self.current_step >= self.max_steps_per_episode

    def _is_target_reached(self):
        return self._get_distance_to_target() < self.target_threshold

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("[DroneEnv] Environment closed.")

# --- Train Function ---
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
            print("\n[Trainer] Training manually interrupted. Saving model...")

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

# --- Main ---
if __name__ == "__main__":
    GUI_ENABLED = True
    MAX_EPISODES = 2
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

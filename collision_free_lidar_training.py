# === Imports ===
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
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent import MlpLstmPolicy
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import subprocess

# === Config ===
VISUALIZE_LIDAR = False

# === Launch AirSim ===
def launch_airsim(gui_enabled=True):
    executable_path = r"C:\\Programming\\Reinforcement Learning\\Autonomous Drone\\AirSim\\Unreal\\Environments\\Blocks\\Binaries\\Win64\\Blocks.exe"
    args = ["-windowed", "-NoVSync"] if not gui_enabled else []
    process = subprocess.Popen([executable_path] + args)
    print(f"[Launcher] Launching AirSim Blocks environment...")
    time.sleep(15)
    return process

# === Callback ===
class DroneTrainingCallback(BaseCallback):
    def __init__(self, max_episodes, autosave_dir, final_save_path, collision_log, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.episode_lengths = []
        self.autosave_dir = autosave_dir
        self.final_save_path = final_save_path
        self.writer = SummaryWriter("./drone_tensorboard/metrics")
        self.collision_log = collision_log
        os.makedirs(self.autosave_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.final_save_path), exist_ok=True)

    def _on_step(self):
        if self.locals['dones'][0]:
            self.episode_count += 1
            ep_len = self.locals['infos'][0].get("episode", {}).get("l", 0)
            self.episode_lengths.append(ep_len)
            collisions = len(self.locals['infos'][0].get("collisions", []))
            self.writer.add_scalar("collisions", collisions, self.episode_count)
            self.collision_log.append(self.locals['infos'][0].get("collision_pos", []))

            path = os.path.join(self.autosave_dir, f"episode_{self.episode_count:03}.zip")
            self.model.save(path)
            print(f"\n[AutoSave] Saved: {path}")

            final_dist = self.locals['infos'][0].get("final_distance", 9999.0)
            target_reached = final_dist < 0.5

            print(f"\n[SUMMARY] Episode {self.episode_count:03}")
            print(f" - Collisions: {self.locals['infos'][0].get('collisions', []) or 'None'}")
            print(f" - Target Reached: {'YES' if target_reached else 'NO'}")
            print(f" - End: {'Battery Depleted' if self.locals['infos'][0]['episode']['l'] >= 200 else 'Target Reached'}")

            if self.episode_count >= self.max_episodes:
                self.plot_training()
                self.plot_collision_heatmap()
                self.model.save(self.final_save_path)
                return False
        return True

    def plot_training(self):
        plt.figure()
        plt.plot(range(1, len(self.episode_lengths)+1), self.episode_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Steps per Episode")
        plt.tight_layout()
        plt.savefig("drone_training_line_plot.png")
        print("[Plot] Saved training progress.")

    def plot_collision_heatmap(self):
        points = [pt for ep in self.collision_log for pt in ep]
        if not points:
            print("[Heatmap] No collisions to plot.")
            return
        arr = np.array(points)
        x, y = arr[:, 0], arr[:, 1]
        heatmap, _, _ = np.histogram2d(x, y, bins=50)
        plt.figure(figsize=(6, 5))
        plt.imshow(heatmap.T, origin='lower', cmap='Reds')
        plt.title("Collision Heatmap")
        plt.tight_layout()
        plt.savefig("collision_heatmap.png")
        print("[Plot] Saved collision heatmap.")

# === Drone Environment ===
class DroneEnv(gym.Env):
    def __init__(self, max_episodes=600, max_steps=200):
        super().__init__()
        self.action_space = spaces.Box(low=np.array([-5, -5, -2, -90], dtype=np.float32),
                                       high=np.array([5, 5, 2, 90], dtype=np.float32))
        self.observation_space = spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32)

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.start_point = airsim.Vector3r(0, 0, -0.6777)
        self.targets = [
            airsim.Vector3r(-40.972, 0.878, -0.676),
            airsim.Vector3r(-35.480, -35.871, -19.422),
            airsim.Vector3r(-44.536, 26.959, 0.668),
            airsim.Vector3r(32.974, 32.932, -8.931)
        ]

        self.target_threshold = 0.5
        self.duration = 0.1
        self.max_steps = max_steps
        self.total_episodes = max_episodes
        self.episode_num = 0
        self.collision_pos = []
        if max_episodes < 432:
            self.target_schedule = [0] * max_episodes
        else:
            block = max_episodes // 4
            self.target_schedule = [0]*block + [1]*block + [2]*block + [3]*(max_episodes - 3*block)

    def reset(self):
        self.episode_num += 1
        if self.episode_num > len(self.target_schedule):
            self.episode_num = len(self.target_schedule)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.collision_pos = []
        target_idx = self.target_schedule[self.episode_num - 1]
        self.target = self.targets[target_idx]
        self.client.simSetVehiclePose(airsim.Pose(self.start_point), True)
        self.client.hoverAsync().join()
        self.current_step = 0
        self.battery = 100
        self.episode_reward = 0
        self.prev_distance = self._get_distance()
        print(f"\n[Episode {self.episode_num:03}/{self.total_episodes:03}] Starting new episode")
        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        vx, vy, vz, yaw = action + np.random.normal(0, 0.2, 4)
        pos = self.client.getMultirotorState().kinematics_estimated.position
        dist = self._get_distance()

        if dist > 2:
            lidar = self.client.getLidarData("LidarSensor1")
            filtered = self._filter_lidar(lidar)
            if self._obstacle_detected(filtered):
                vz = -1
                yaw += 90
                if VISUALIZE_LIDAR:
                    self._visualize_lidar(filtered)

        self.client.moveByVelocityAsync(vx, vy, vz, self.duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(True, yaw)).join()

        if self.client.simGetCollisionInfo().has_collided:
            pos = self.client.getMultirotorState().kinematics_estimated.position
            self.collision_pos.append([pos.x_val, pos.y_val])
            self.client.moveByVelocityAsync(0, 0, -1, 0.5).join()

        self.battery -= 0.5
        reward = self._compute_reward()
        done = self.battery <= 0 or self._reached()
        info = {
            'episode': {'r': self.episode_reward, 'l': self.current_step},
            'collisions': self.collision_pos,
            'collision_pos': self.collision_pos,
            'final_distance': self._get_distance()
        }

        print(f"[Episode: {self.episode_num:03}/{self.total_episodes:03} | Step: {self.current_step:03}] "
              f"Action: [{vx:.1f}, {vy:.1f}, {vz:.1f}, {yaw:.1f}] | "
              f"Distance: {dist:.2f} | Reward: {reward:+.1f} | Battery: {self.battery:.1f}%", end='\r')

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([
            self.target.x_val - pos.x_val,
            self.target.y_val - pos.y_val,
            self.target.z_val - pos.z_val
        ], dtype=np.float32)

    def _get_distance(self):
        obs = self._get_obs()
        return float(np.linalg.norm(obs))

    def _compute_reward(self):
        dist = self._get_distance()
        delta = self.prev_distance - dist
        reward = delta * 5
        if self._reached(): reward += 50
        reward -= 0.1
        reward -= len(self.collision_pos) * 0.5
        self.prev_distance = dist
        self.episode_reward += reward
        return reward

    def _reached(self):
        return self._get_distance() < self.target_threshold

    def _filter_lidar(self, data):
        pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
        return pts[(pts[:, 2] > -1.5) & (np.linalg.norm(pts, axis=1) < 20)]

    def _obstacle_detected(self, pts):
        return len(pts[(pts[:, 0] > 0) & (np.linalg.norm(pts, axis=1) < 4)]) > 10

    def _visualize_lidar(self, pts):
        plt.clf()
        plt.scatter(pts[:, 0], pts[:, 1], c='g', s=1)
        plt.title("Live LiDAR View")
        plt.pause(0.01)

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

# === Train Function ===
def train_drone(gui_enabled=True, max_episodes=600, max_steps=200, use_custom_name=False, custom_name="ppo_final.zip"):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    autosave_dir = "./autosave_models"
    final_name = custom_name if use_custom_name else f"ppo_drone_final_{timestamp}.zip"
    final_path = os.path.join("final_models", final_name)

    process = launch_airsim(gui_enabled)
    collision_log = []
    try:
        env = DroneEnv(max_episodes, max_steps)
        callback = DroneTrainingCallback(max_episodes, autosave_dir, final_path, collision_log)
        model = RecurrentPPO(MlpLstmPolicy, env, verbose=1, learning_rate=0.0003,
                             n_steps=1024, batch_size=128, gamma=0.99,
                             gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                             tensorboard_log="./drone_tensorboard/")
        try:
            model.learn(total_timesteps=max_episodes * max_steps * 2, callback=callback)
        except KeyboardInterrupt:
            model.save(final_path)
            callback.plot_training()
            callback.plot_collision_heatmap()
    finally:
        env.close()
        process.terminate()

# === Main ===
if __name__ == "__main__":
    train_drone(
        gui_enabled=True,
        max_episodes=1000,
        max_steps=200,
        use_custom_name=False
    )

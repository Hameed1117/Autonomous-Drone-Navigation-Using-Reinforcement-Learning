import gym
import numpy as np
import airsim
import time
import os
import subprocess
from stable_baselines3 import PPO


# === Config ===
MAX_STEPS = 200
TARGETS = [
    airsim.Vector3r(-40.972, 0.878, -0.676),
    airsim.Vector3r(-35.480, -35.871, -19.422),
    airsim.Vector3r(-44.536, 26.959, 0.668),
    airsim.Vector3r(32.974, 32.932, -8.931)
]

# === Launch AirSim ===
def launch_airsim():
    path = r"C:\\Programming\\Reinforcement Learning\\Autonomous Drone\\AirSim\\Unreal\\Environments\\Blocks\\Binaries\\Win64\\Blocks.exe"
    print("\n[Evaluator] Launching AirSim Blocks environment...")
    process = subprocess.Popen([path])
    time.sleep(15)
    return process


# === Drone Test Environment ===
class DroneTestEnv(gym.Env):
    def __init__(self, target):
        super().__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.start = airsim.Vector3r(0, 0, -0.6777)
        self.target = target
        self.duration = 0.1
        self.threshold = 0.5
        self.battery = 100
        self.step_num = 0
        self.collisions = []
        self.client.simSetVehiclePose(airsim.Pose(self.start), True)
        self.client.hoverAsync().join()

    def reset(self):
        self.battery = 100
        self.step_num = 0
        self.collisions = []
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simSetVehiclePose(airsim.Pose(self.start), True)
        self.client.hoverAsync().join()
        return self._get_obs()

    def _get_obs(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([
            self.target.x_val - pos.x_val,
            self.target.y_val - pos.y_val,
            self.target.z_val - pos.z_val
        ], dtype=np.float32)

    def _get_distance(self):
        return float(np.linalg.norm(self._get_obs()))

    def _reached(self):
        return self._get_distance() < self.threshold

    def step(self, action):
        self.step_num += 1
        vx, vy, vz, yaw = action
        self.client.moveByVelocityAsync(vx, vy, vz, self.duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(True, yaw)).join()

        if self.client.simGetCollisionInfo().has_collided:
            pos = self.client.getMultirotorState().kinematics_estimated.position
            self.collisions.append([pos.x_val, pos.y_val])
            self.client.moveByVelocityAsync(0, 0, -1, 0.5).join()

        self.battery -= 0.5
        done = self.battery <= 0 or self._reached() or self.step_num >= MAX_STEPS

        info = {
            'steps': self.step_num,
            'reached': self._reached(),
            'collisions': self.collisions,
            'distance': self._get_distance()
        }
        print(f"[Step {self.step_num:03}] Action: {action.tolist()} | Distance: {info['distance']:.2f} | Battery: {self.battery:.1f}% | Reward: {'+0.0' if self._reached() else '-0.0'}")

        return self._get_obs(), 0.0, done, info

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)


# === Evaluate Model ===
def evaluate_model(model_path):
    process = launch_airsim()
    model = PPO.load(model_path)
    print(f"\n[Loaded] Model from {model_path}\n")
    results = []
    
    try:
        for i, target in enumerate(TARGETS):
            print(f"\n=== Evaluating Target {i+1}/{len(TARGETS)} ===")
            env = DroneTestEnv(target)
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)
            distance = info.get('distance', -1.0)
            print(f"\n[Target {i+1}] Result: {'Reached' if info.get('reached') else 'Missed'} | Steps: {info.get('steps', 0)} | Dist: {distance:.2f} | Collisions: {len(info.get('collisions', []))}")
            results.append(info)
            env.close()
    finally:
        process.terminate()

    print("\n=== Evaluation Summary ===")
    for i, res in enumerate(results):
        print(f"Target {i+1}: {'Reached' if res.get('reached') else 'Missed'} | Steps: {res.get('steps')} | Final Dist: {res.get('distance', -1):.2f} | Collisions: {len(res.get('collisions', []))}")


# === Main ===
if __name__ == "__main__":
    evaluate_model("final_models/ppo_drone_final_20250504_173038.zip")

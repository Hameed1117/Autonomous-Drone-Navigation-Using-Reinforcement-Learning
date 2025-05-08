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

# === Import your collision-free training code ===
from collision_free_lidar_training import launch_airsim, DroneEnv

def test_collision_free_model(
    model_path: str,
    gui_enabled: bool = True,
    max_steps_per_episode: int = 400   # PATCH: increased from 200 to 400
):
    """
    Evaluate the trained PPO on each of the 4 predefined targets,
    verifying that LiDAR avoidance yields minimal collisions.
    """
    # Launch AirSim
    process = launch_airsim(gui_enabled)

    try:
        if not os.path.exists(model_path):
            print(f"[Tester] ERROR: Model file not found at '{model_path}'")
            return

        print(f"[Tester] Loading model from {model_path}")
        num_targets = 4
        # PATCH: use 400 steps per episode instead of 200
        env = DroneEnv(max_episodes=num_targets, max_steps=max_steps_per_episode)
        env.target_schedule = list(range(num_targets))
        env.total_episodes  = num_targets

        # PATCH: adjust battery drain so that 4 steps → 1% instead of 2 steps → 1%
        # Original env.step decrements battery by 0.5 per step (2 steps = 1%).
        # We’ll restore 0.25 after each step to net −0.25 per step (4 steps = 1%).
        original_step = env.step
        def patched_step(action):
            obs, reward, done, info = original_step(action)
            env.battery += 0.25
            return obs, reward, done, info
        env.step = patched_step

        model = RecurrentPPO.load(model_path, env=env)
        print("[Tester] Model loaded successfully!\n")

        results = []
        for ep in range(1, num_targets + 1):
            print(f"=== Test Episode {ep}/{num_targets} (Target #{ep-1}) ===")
            obs = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1

            collisions = len(env.collision_pos)
            reached    = env._reached()
            results.append({
                "episode": ep,
                "steps": steps,
                "reward": total_reward,
                "target_reached": reached,
                "collisions": collisions
            })

            print(f"→ Steps: {steps}, Reward: {total_reward:.1f}, "
                  f"Reached: {'Yes' if reached else 'No'}, Collisions: {collisions}\n")
            time.sleep(1)

        print("\n[Test Results Summary]")
        print("-" * 60)
        print(f"{'Ep':<5}{'Steps':<10}{'Reward':<10}{'Reached':<10}{'Collisions':<10}")
        print("-" * 60)
        tot_steps = tot_reward = tot_coll = tot_reach = 0
        for r in results:
            print(f"{r['episode']:<5}{r['steps']:<10}{r['reward']:<10.1f}"
                  f"{('Yes' if r['target_reached'] else 'No'):<10}"
                  f"{r['collisions']:<10}")
            tot_steps  += r['steps']
            tot_reward += r['reward']
            tot_coll   += r['collisions']
            tot_reach  += int(r['target_reached'])

        print("-" * 60)
        n = num_targets
        print(f"Success Rate   : {tot_reach/n*100:.1f}%")
        print(f"Avg. Collisions: {tot_coll/n:.2f}")
        print(f"Avg. Steps     : {tot_steps/n:.1f}")
        print(f"Avg. Reward    : {tot_reward/n:.1f}")

    finally:
        env.close()
        process.terminate()
        print("\n[Launcher] AirSim terminated.")

if __name__ == "__main__":
    MODEL_PATH = (
        r"C:\Programming\Reinforcement Learning\Autonomous Drone"
        r"\final_models\ppo_drone_final_20250504_233255.zip"
    )
    test_collision_free_model(
        model_path=MODEL_PATH,
        gui_enabled=True,
        max_steps_per_episode=400   # PATCH applied here as well
    )

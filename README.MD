# RL Drone Agent

A project for simulating and training an autonomous drone using Microsoft AirSim and reinforcement learning (RL). This repository includes code for:
- Basic drone simulation
- Manual control using keyboard inputs
- A random navigation agent
- Training an RL agent using PPO

Each Python script (except `setup_airsim.py`) has an accompanying video demonstration accessible via cloud storage links.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Setup AirSim](#setup-airsim)
  - [Drone Simulation](#drone-simulation)
  - [Manual Control](#manual-control)
  - [Random Agent](#random-agent)
  - [RL Training](#rl-training)
- [Demonstration Videos](#demonstration-videos)
- [Progress & Next Steps](#progress--next-steps)
- [License](#license)

## Overview

This repository demonstrates the use of Microsoft AirSim in combination with reinforcement learning to develop an autonomous drone agent. The project includes:
- **Drone Simulation:** Automatically runs a simple drone simulation.
- **Manual Control:** Allows keyboard-based control of the drone.
- **Random Agent:** Uses random commands for basic navigation.
- **RL Training:** Implements a custom Gym environment and trains an RL agent (using PPO) to navigate toward a target.

## Project Structure

```
RL_Drone_Agent/
├── .gitignore
├── README.md
├── setup_airsim.py            # Script to configure AirSim settings (no video demonstration)
├── drone_simulation.py        # Automatic drone simulation script
├── manual_control_drone.py    # Manual control via keyboard inputs
├── random_agent_drone.py      # Random navigation agent
├── train_drone_rl.py          # RL training script using PPO
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone git@github.com:Hameed1117/RL_Drone_Agent.git
   cd RL_Drone_Agent
   ```

2. **Set Up the Virtual Environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install airsim numpy gym stable-baselines3 tqdm keyboard
   ```

4. **Configure AirSim Settings:**
   
   Create or update the AirSim settings file at `%LOCALAPPDATA%\AirSim\settings.json` with at least:
   ```json
   {
     "SettingsVersion": 1.2,
     "SimMode": "Multirotor"
   }
   ```

5. **Build the Blocks Environment:**
   
   Follow the AirSim build instructions to build the Blocks environment with Unreal Engine. Ensure the executable path in the scripts is correct.

## Usage

**Note:** Run all scripts with Administrator privileges for proper AirSim API access.

### Setup AirSim

Before running any simulation, if needed, run:
```bash
python setup_airsim.py
```

### Drone Simulation

Run the automatic drone simulation:
```bash
python drone_simulation.py
```

### Manual Control

Control the drone manually using keyboard inputs:
```bash
python manual_control_drone.py
```

### Random Agent

Run the random navigation agent:
```bash
python random_agent_drone.py
```

### RL Training

Train the RL agent using PPO:
```bash
python train_drone_rl.py
```

The training script launches the Blocks environment automatically, displays a progress bar in the terminal, and saves the trained model as `drone_ppo_model.zip`.

## Demonstration Videos

Below are the demonstration videos for each Python script (except `setup_airsim.py`), available on Google Drive:

- **Basic Drone Simulation**: [Watch Demo](https://drive.google.com/file/d/1kNGf02cGb-eNwWXVZntePsQn2O5faXfE/view?usp=sharing)
- **Manual Control Demo**: [Watch Demo](https://drive.google.com/file/d/1r3-a54kx2OKara78LTqwPXHIaU39sPqb/view?usp=sharing)
- **Random Agent Navigation**: [Watch Demo](https://drive.google.com/file/d/1moeeI8EAaNKIiVbFId0W21O__Fxp6lyf/view?usp=sharing)
- **RL Training Process**: [Watch Demo](https://drive.google.com/file/d/1lBa5Ivnq9UHrEXbAAtZIdY8lVeZt617o/view?usp=sharing)

(Replace XXXX, YYYY, ZZZZ, and AAAA with your actual Google Drive file IDs)

## Progress & Next Steps

### What We Have Done

**Environment Setup:**
- Configured Microsoft AirSim and built the Blocks environment for multirotor simulation.

**Manual & Random Control:**
- Developed scripts for manual control and random agent navigation to verify API connectivity.

**Custom Gym Environment:**
- Wrapped AirSim in a custom Gym environment (DroneEnv) for RL integration.

**RL Training Integration:**
- Implemented RL training using PPO with real-time progress tracking and visual feedback in the simulation.

### Challenges Faced

**False Collision Detection:**
- AirSim sometimes reported collisions falsely. Various thresholds and debugging outputs were added to address this.

**Altitude and Action Control:**
- Tuning the drone's altitude and movement was challenging. We eventually opted for a fixed altitude for RL training.

**Visualization:**
- Ensuring a single, clear target marker and avoiding multiple overlays required iterative refinements.

### Next Steps

**Reward Function Refinement:**
- Optimize the reward function for smoother navigation.

**Extended Training:**
- Increase the number of training timesteps for more robust learning.

**Sensor Integration:**
- Integrate additional sensor data (e.g., camera, lidar) for improved state representation.

**Sim-to-Real Transfer:**
- Explore strategies to transfer the learned policy to a real drone.

**Enhanced Visualization:**
- Improve real-time visualization and debugging outputs in the simulation.

## License

[MIT License]

For any issues or further modifications, please open an issue or contact the repository maintainer.
RL Drone Agent
This repository contains code for simulating and training an autonomous drone using Microsoft AirSim and reinforcement learning (RL). The project includes scripts for:

Setting up AirSim
Running an automatic drone simulation
Manual control of the drone via keyboard inputs
A random agent for drone navigation
Training an RL agent using PPO (Proximal Policy Optimization)
Table of Contents
Overview
Requirements
Installation
How to Run the Code
Project Structure
Progress and Challenges
Next Steps
Overview
We set up Microsoft AirSim on a Windows PC with the Blocks environment (an Unreal Engine simulation) to simulate a drone. A custom Gym environment (DroneEnv) was developed to interface with AirSim, allowing reinforcement learning with stable-baselines3. We implemented several scripts:

Manual control: Allows real-time drone control using the keyboard.
Random agent: Sends random commands to the drone to navigate toward a target.
RL training: Uses PPO to train the drone to navigate to a target (visualized as a red marker in the Blocks environment).
Requirements
Operating System: Windows 10 or higher
Python: 3.8+
Microsoft AirSim: GitHub - microsoft/AirSim
Unreal Engine: (for building the Blocks environment)
Visual Studio: With C++ development tools
Git
Python Dependencies
Install the following packages:

airsim
numpy
gym
stable-baselines3
tqdm
keyboard (for manual control)
You can install these via pip. If a requirements.txt is available, run:

bash
Copy
pip install -r requirements.txt
Otherwise, install manually:

bash
Copy
pip install airsim numpy gym stable-baselines3 tqdm keyboard
Installation
Clone the Repository:

bash
Copy
git clone git@github.com:Hameed1117/RL_Drone_Agent.git
cd RL_Drone_Agent
Set Up the Virtual Environment:

bash
Copy
python -m venv .venv
.venv\Scripts\activate  # On Windows
Install Python Dependencies:

bash
Copy
pip install -r requirements.txt
Configure AirSim Settings:

Ensure that the AirSim settings file exists at %LOCALAPPDATA%\AirSim\settings.json with the following minimal content:

json
Copy
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor"
}
Build the Blocks Environment:

Follow the AirSim build instructions to build the Blocks environment with Unreal Engine. Ensure the executable exists at the path specified in the scripts (update the path in the code if necessary).

How to Run the Code
Note: It is recommended to run these scripts with Administrator privileges for proper API access.

1. Setup AirSim
Run the setup script (if needed):

bash
Copy
python setup_airsim.py
2. Automatic Drone Simulation
To run a basic automatic simulation of the drone:

bash
Copy
python drone_simulation.py
3. Manual Control of the Drone
To control the drone manually using keyboard inputs:

bash
Copy
python manual_control_drone.py
4. Random Agent Simulation
To run the random agent that sends random commands to the drone:

bash
Copy
python random_agent_drone.py
5. RL Training with PPO
To train the RL agent (which navigates toward a target marker) using PPO:

bash
Copy
python train_drone_rl.py
The training script automatically launches the Blocks environment, displays a progress bar in the terminal (via Tqdm), and saves the trained model as drone_ppo_model.zip in the repository.

Project Structure
Copy
RL_Drone_Agent/
├── .gitignore
├── drone_simulation.py
├── manual_control_drone.py
├── random_agent_drone.py
├── setup_airsim.py
├── train_drone_rl.py
└── README.md
Note: The .gitignore is configured to track only .py files.

Progress and Challenges
Progress
AirSim Setup:
Successfully configured AirSim with the Blocks environment for multirotor simulation.

Manual Control:
Developed a manual control script to directly control the drone using keyboard inputs, confirming API connectivity.

Random Agent:
Implemented a random agent that navigates toward a target, with debugging to filter out false collision signals.

Custom Gym Environment:
Wrapped the AirSim simulation in a custom Gym environment (DroneEnv) to interface with reinforcement learning algorithms.

RL Training Integration:
Integrated PPO from stable-baselines3 with the custom environment, including visualization of a target marker in the simulation and a Tqdm progress bar for training.

Challenges
Dependency & API Connection Issues:
Initial challenges included installing dependencies (e.g., numpy, msgpackrpc) and handling connection errors with AirSim’s API.

False Collision Detection:
AirSim reported collisions (often with the ground) even when the drone did not physically contact any obstacles. We attempted to mitigate this by analyzing impact points and applying thresholds.

Altitude and Action Control:
The drone would sometimes descend or reset prematurely due to vertical control issues. We experimented with fixing the altitude by removing vertical control (forcing vz = 0) so that the drone maintains a constant altitude (–10 m).

Visualization:
Ensuring only a single, large target marker is visible in the simulation required iterative refinements with the simPlotPoints API.

Episode Termination Conditions:
Fine-tuning conditions (collision, target reached, maximum steps) to prevent premature resets was challenging.

Next Steps
Reward Function Optimization:
Further refine the reward structure (e.g., penalizing excessive turning or lateral drift) for improved agent performance.

Extended Training:
Increase training timesteps and evaluate model performance across different target locations and scenarios.

Sensor Data Integration:
Enhance state representation by incorporating additional sensor data (e.g., camera, lidar) for more robust navigation.

Sim-to-Real Transfer:
Investigate methods to transfer the trained RL policy from simulation to an actual drone.

Enhanced Debugging & Visualization:
Consider real-time trajectory visualization, logging of additional metrics (e.g., reward trends), and integration with TensorBoard.
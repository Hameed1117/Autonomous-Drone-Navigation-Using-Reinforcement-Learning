# Autonomous Drone with Reinforcement Learning

A comprehensive project for training and testing autonomous drone navigation using Microsoft AirSim and Reinforcement Learning. This repository implements a PPO (Proximal Policy Optimization) agent that learns to navigate to specific targets in a simulated environment.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Setup AirSim](#setup-airsim)
  - [Manual Control](#manual-control)
  - [Training the Agent](#training-the-agent)
  - [Testing the Agent](#testing-the-agent)
  - [Custom Target Testing](#custom-target-testing)
- [Key Features](#key-features)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Overview

This project demonstrates autonomous drone navigation using reinforcement learning techniques. The main components include:

- **Environment Setup**: Integration with Microsoft AirSim simulator for realistic drone physics
- **Manual Control**: Script for keyboard-based drone control for testing and data collection
- **RL Training**: Custom Gym environment and PPO implementation to train a navigation agent
- **Testing Framework**: Comprehensive testing tools to evaluate agent performance on various targets
- **Visualization**: Enhanced visual indicators for targets and navigation paths

## Project Structure

```
Autonomous Drone/
├── .gitignore
├── README.md
├── manual_control_drone.py    # Manual drone control script
├── setup_airsim.py            # AirSim configuration script
├── training.py                # Main PPO agent training script
├── test_drone.py              # Basic model testing script
├── test_drone_custom_targets.py # Advanced testing with custom targets
├── drone_model.zip            # Saved trained model
└── requirements.txt           # Project dependencies
```

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Autonomous\ Drone
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AirSim**:
   Follow the AirSim documentation to install and configure AirSim with Unreal Engine. The Blocks environment is recommended for this project.

## Usage

### Setup AirSim

Before running any simulation, configure the AirSim settings:
```bash
python setup_airsim.py
```

### Manual Control

Test the drone with manual keyboard controls:
```bash
python manual_control_drone.py
```

### Training the Agent

Train the PPO agent to navigate to a target:
```bash
python training.py
```

The script will:
- Automatically launch the AirSim Blocks environment
- Create a visual marker at the target location
- Train the PPO agent for the specified number of episodes
- Save the trained model as `drone_model.zip`

### Testing the Agent

Test the trained model on the original training target:
```bash
python test_drone.py
```

This will evaluate the model's performance and provide a detailed summary of results.

### Custom Target Testing

Test how well the agent generalizes to new target locations:
```bash
python test_drone_custom_targets.py
```

This advanced testing script:
- Tests the agent on multiple target locations including the original and custom targets
- Adds clear visual markers in the environment
- Provides comprehensive performance metrics for each target

## Key Features

- **Reinforcement Learning Implementation**: Custom Gym environment integrated with AirSim for PPO training
- **Realistic Simulation**: Uses AirSim's physics engine for realistic drone dynamics
- **Visual Feedback**: Enhanced visual indicators for targets and navigation paths
- **Comprehensive Testing**: Tools to evaluate agent generalization to unseen targets
- **Progress Tracking**: Detailed performance metrics during training and testing
- **Reward Mechanism**: Distance-based reward system to encourage efficient navigation

## Results

Our trained PPO agent demonstrates:
- **100% success rate** on the original training target
- **Consistent navigation path** with approximately 108 steps to reach the target
- **Efficient energy usage** (less than 54% battery consumption per mission)
- **Variable performance** on novel targets depending on distance and direction

## Future Improvements

- **Obstacle Avoidance**: Enhance the environment to include obstacles and train the agent to avoid them
- **Multi-Target Navigation**: Extend the agent to handle sequences of waypoints
- **Dynamic Rewards**: Implement more sophisticated reward functions based on energy efficiency and path smoothness
- **Camera Integration**: Use drone camera input as additional state information
- **Transfer Learning**: Apply techniques to transfer the learned policy to real-world drones

## Author

Khadhar Hameed Khan Pathan - @Hameed1117
# Autonomous Drone - PPO-Based RL Training

This branch implements reinforcement learning for autonomous drone navigation using **Proximal Policy Optimization (PPO)** in a simulated environment powered by **Microsoft AirSim**.

---

## 🎯 Project Objective

Train a drone to autonomously navigate to target locations within a 3D simulated space using PPO. The drone receives distance-based rewards and learns efficient paths over training episodes.

---

## 📁 Project Structure

```plaintext
Autonomous Drone/
├── final_models/                      # All saved PPO models
│   ├── ppo_drone_final_20250504_173038.zip
│   ├── ppo_drone_final_20250504_233255.zip
│   └── ppo_drone_final_20250507_221315.zip
├── drone_model.zip                   # Default trained model
├── drone_training_line_plot.png     # PPO training reward curve
├── collision_heatmap.png            # (For future use with LiDAR branch)
├── manual_control_drone.py          # Keyboard-based drone control
├── setup_airsim.py                  # Configure AirSim settings
├── training.py                      # PPO training script
├── test_drone.py                    # Testing on training target
├── test_drone_custom_targets.py     # Testing on custom targets
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Autonomous-Drone
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 🔧 Step 1: Configure AirSim

Ensure Microsoft AirSim is installed and configured correctly. Use the Blocks environment.

```bash
python setup_airsim.py
```

### 🎮 Step 2: Manual Drone Control

Fly the drone manually using keyboard input:

```bash
python manual_control_drone.py
```

### 🧠 Step 3: Train the PPO Agent

Start training using PPO:

```bash
python training.py
```

* Automatically launches AirSim
* Initializes target location
* Trains PPO agent
* Saves model to `/final_models/` and `drone_model.zip`
* Logs training progress and reward history

### ✅ Step 4: Test the Agent (Original Target)

```bash
python test_drone.py
```

### 🌍 Step 5: Custom Target Evaluation

```bash
python test_drone_custom_targets.py
```

* Tests on unseen target positions
* Evaluates generalization ability

---

## 📊 Results

* 🏁 Success rate on training target: **100%**
* ⚡ Avg. steps to reach target: \~108
* 🔋 Avg. battery use per mission: \~54%
* 📈 Training progress plotted in `drone_training_line_plot.png`

---

## 🗂️ Saved PPO Models

Models saved during/after training are found in:

```
/final_models/
```

Example:

* `ppo_drone_final_20250507_221315.zip` — Latest stable checkpoint
* `drone_model.zip` — Default model used for testing scripts

---

## 🔮 Future Branches

The next development branch will focus on:

* LiDAR-based collision avoidance
* Recurrent PPO (LSTM policy)
* Obstacle-aware path planning

---

## 👤 Author

**Khadhar Hameed Khan Pathan**
GitHub: [@Hameed1117](https://github.com/Hameed1117)

```

---



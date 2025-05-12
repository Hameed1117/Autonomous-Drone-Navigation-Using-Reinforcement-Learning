# Collision-Free Autonomous Drone Navigation with Recurrent PPO

This branch builds on top of the PPO-based drone navigation system by incorporating **LiDAR-based obstacle detection** and training the agent with **Recurrent Proximal Policy Optimization (Recurrent PPO)** to enable safe, collision-free flight in complex environments simulated in Microsoft AirSim.

---

## 🛰️ Objective

Train an autonomous drone to **navigate to target locations while avoiding collisions** using real-time LiDAR sensing and memory-enhanced reinforcement learning. The Recurrent PPO agent leverages past observations to improve path decisions in the presence of obstacles.

---

## 🔧 Features

- ✅ **LiDAR sensor** integration for real-time 360° obstacle detection
- ✅ **Recurrent PPO** agent (LSTM-based policy) using SB3-Contrib
- ✅ **Collision heatmap logging** and step-wise reward shaping
- ✅ **Manual and automated evaluation** on fixed/custom target locations
- ✅ **Obstacle-aware reward design** with scan-pause-turn-flight logic

---

## 📁 Directory Structure

```plaintext
Autonomous Drone/
├── final_models/
│   ├── ppo_drone_final_20250504_173038.zip
│   ├── ppo_drone_final_20250504_233255.zip
│   └── ppo_drone_final_20250507_221315.zip
├── collision_free_lidar_training.py        # Training with LiDAR and Recurrent PPO
├── collision_free_lidar_testing.py         # Basic model testing on original targets
├── collision_free_manual_target_test.py    # Custom target evaluation with obstacles
├── collision_heatmap.png                   # Heatmap showing collision-prone zones
├── manual_control_drone.py                 # Optional: Manual testing tool
├── requirements.txt                        # Python dependencies
├── setup_airsim.py                         # AirSim environment settings
├── README.md                               # This file
````

---

## 🛠️ Setup Instructions

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

### 📦 Step 1: AirSim Configuration

Make sure you’ve installed Microsoft AirSim with LiDAR enabled. Use the Blocks environment for best compatibility.

```bash
python setup_airsim.py
```

### 🧠 Step 2: Start Recurrent PPO Training

```bash
python collision_free_lidar_training.py
```

This script will:

* Launch AirSim
* Train the agent using Recurrent PPO and LiDAR
* Log metrics and save model to `/final_models/`
* Output heatmap of risky zones

### ✅ Step 3: Test on Default Scenario

```bash
python collision_free_lidar_testing.py
```

### 🌍 Step 4: Test on Custom Targets

```bash
python collision_free_manual_target_test.py
```

---

## 📊 Performance Metrics

* 🧠 **Recurrent PPO Agent** trained successfully to navigate around static obstacles
* ✅ Demonstrated collision-free arrival at training targets
* 🧪 **Generalization** observed with varied success on custom unseen targets
* 🔋 Energy efficiency preserved by smarter trajectory planning

---

## 📌 Notes

* Models are stored in `final_models/`
* `collision_heatmap.png` shows the zones with frequent collisions early in training
* Further tuning and curriculum learning will be handled in a future branch

---

## 🧭 Next Steps

* ✅ Sensor fusion with drone camera + LiDAR
* ✅ Sequential waypoint navigation
* 🔄 Curriculum-based training environments
* 🔍 Add A\* or RRT-based fallback planner for complex layouts

---

## 👤 Author

**Khadhar Hameed Khan Pathan**
GitHub: [@Hameed1117](https://github.com/Hameed1117)

```

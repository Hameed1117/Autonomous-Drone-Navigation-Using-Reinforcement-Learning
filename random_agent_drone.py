import subprocess
import time
import airsim
import random
import math


def launch_airsim():
    # Update the executable_path to your Blocks.exe location if needed.
    executable_path = r"C:\Programming\Reinforcement Learning\Autonomous Drone\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe"
    process = subprocess.Popen(executable_path)
    print("[Launcher] Launching AirSim Blocks environment...")
    # Wait enough time for the simulation to fully load.
    time.sleep(30)
    print("[Launcher] Blocks environment should be up now.")
    return process


def random_agent():
    # Connect to AirSim.
    client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
    client.confirmConnection()
    print("[API] Connected to AirSim.")

    client.enableApiControl(True)
    client.armDisarm(True)
    print("[API] Drone armed and under API control.")

    # Take off and reach a safe altitude.
    print("[Action] Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)

    # Define fixed start and target points (NED coordinates).
    # Adjust these points if necessary to be in open areas.
    point_A = airsim.Vector3r(30, 10, -10)
    point_B = airsim.Vector3r(20, 20, -10)
    target_threshold = 5.0  # meters within target to count as reached

    # Move to start point A.
    print(f"[Action] Moving to start point A: ({point_A.x_val}, {point_A.y_val}, {point_A.z_val})")
    client.moveToPositionAsync(point_A.x_val, point_A.y_val, point_A.z_val, 5).join()
    time.sleep(2)

    print(f"[Action] Starting random navigation towards point B: ({point_B.x_val}, {point_B.y_val}, {point_B.z_val})")

    while True:
        # Check for collision.
        collision_info = client.simGetCollisionInfo()
        if collision_info.has_collided:
            print("[Alert] Collision detected! Ending simulation.")
            break

        # Get current position.
        current_pose = client.simGetVehiclePose()
        pos = current_pose.position
        distance = math.sqrt((pos.x_val - point_B.x_val) ** 2 +
                             (pos.y_val - point_B.y_val) ** 2 +
                             (pos.z_val - point_B.z_val) ** 2)
        print(
            f"[Status] Current position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f}), Distance to target: {distance:.2f}")

        # Check if target reached.
        if distance < target_threshold:
            print("[Success] Target reached successfully!")
            break

        # Check if drone has landed prematurely (z > -1 means near ground).
        if pos.z_val > -1:
            print("[Alert] Drone has landed prematurely. Ending simulation.")
            break

        # Generate random velocities with reduced magnitude.
        vx = random.uniform(-2, 2)  # forward/backward velocity in m/s
        vy = random.uniform(-2, 2)  # lateral velocity in m/s
        vz = random.uniform(-0.2, 0.2)  # vertical velocity in m/s (negative: ascend)
        duration = 1.0  # command duration in seconds

        print(f"[Action] Random command: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f} for {duration} sec")
        client.moveByVelocityAsync(vx, vy, vz, duration).join()

    # End simulation: hover and land.
    print("[Action] Hovering and landing...")
    client.hoverAsync().join()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("[API] Simulation ended.")


def main():
    sim_process = launch_airsim()
    try:
        random_agent()
    finally:
        print("[Launcher] Terminating simulation...")
        sim_process.terminate()


if __name__ == '__main__':
    main()
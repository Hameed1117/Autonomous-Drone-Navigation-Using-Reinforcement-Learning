import subprocess
import time
import airsim


def launch_airsim():
    # Update the executable_path to the location of your built Blocks.exe.
    executable_path = r"C:\Programming\Reinforcement Learning\Autonomous Drone\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe"
    process = subprocess.Popen(executable_path)
    print("Launching AirSim Blocks environment...")
    # Wait enough time (e.g., 20-30 seconds) for the simulation to load.
    time.sleep(10)
    return process


def drone_control():
    client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
    client.confirmConnection()
    print("Connected to AirSim.")

    client.enableApiControl(True)
    client.armDisarm(True)
    print("Drone armed and under API control.")

    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)

    print("Moving forward...")
    client.moveByVelocityAsync(5, 0, 0, 5).join()

    print("Hovering...")
    client.hoverAsync().join()
    time.sleep(2)

    print("Landing...")
    client.landAsync().join()

    client.armDisarm(False)
    client.enableApiControl(False)
    print("Drone landed and disarmed.")


def main():
    # Launch the simulation.
    sim_process = launch_airsim()

    try:
        # Run the drone control commands.
        drone_control()
    finally:
        print("Terminating simulation...")
        sim_process.terminate()


if __name__ == '__main__':
    main()

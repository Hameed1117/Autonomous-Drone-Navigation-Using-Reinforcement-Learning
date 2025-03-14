import subprocess
import time
import airsim
import keyboard  # pip install keyboard

def launch_airsim():
    # Update this path to your Blocks.exe location if needed.
    executable_path = r"C:\Programming\Reinforcement Learning\Autonomous Drone\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe"
    process = subprocess.Popen(executable_path)
    print("Launching AirSim Blocks environment...")
    # Wait enough time for the simulation to fully load.
    time.sleep(30)
    return process

def manual_control():
    # Connect to the Blocks environment.
    client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
    client.confirmConnection()
    print("Connected to AirSim.")

    client.enableApiControl(True)
    client.armDisarm(True)
    print("Drone armed and under API control.")

    # Take off.
    print("Taking off...")
    client.takeoffAsync().join()
    time.sleep(2)

    # Instructions for manual control.
    print("Manual control active. Use the following keys:")
    print("  Up Arrow    : Ascend")
    print("  Down Arrow  : Descend")
    print("  W           : Move forward (in the direction drone is facing)")
    print("  S           : Move backward")
    print("  A           : Yaw left")
    print("  D           : Yaw right")
    print("  ESC         : Exit manual control and land")

    try:
        while True:
            # Exit manual control when ESC is pressed.
            if keyboard.is_pressed("esc"):
                print("ESC pressed. Exiting manual control loop.")
                break

            # Duration for each command.
            duration = 0.1  # seconds

            # Initialize command values.
            vx = 0    # forward/backward (body frame)
            vz = 0    # vertical (NED: negative = up)
            yaw_rate = 0  # yaw rate in degrees per second

            # Vertical control (arrow keys).
            if keyboard.is_pressed("up"):
                vz = -2   # ascend (negative in NED)
            elif keyboard.is_pressed("down"):
                vz = 2    # descend

            # Forward/backward in the body frame.
            if keyboard.is_pressed("w"):
                vx = 5    # move forward
            elif keyboard.is_pressed("s"):
                vx = -5   # move backward

            # Yaw control (rotate drone).
            if keyboard.is_pressed("a"):
                yaw_rate = -30  # rotate left
            elif keyboard.is_pressed("d"):
                yaw_rate = 30   # rotate right

            # Create a yaw_mode object.
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)

            # Send the command using the drone's body frame.
            client.moveByVelocityBodyFrameAsync(vx, 0, vz, duration, yaw_mode=yaw_mode)
            time.sleep(duration)

    except Exception as e:
        print("Error during manual control:", e)

    finally:
        # When the manual control loop exits, hover and land.
        print("Exiting manual control. Hovering and landing...")
        client.hoverAsync().join()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        print("Drone landed and disarmed.")

def main():
    # Launch the Blocks environment.
    sim_process = launch_airsim()
    try:
        manual_control()
    finally:
        print("Terminating simulation...")
        sim_process.terminate()

if __name__ == '__main__':
    main()

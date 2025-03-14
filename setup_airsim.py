import os
import json


def write_settings():
    local_appdata = os.getenv('LOCALAPPDATA')
    if not local_appdata:
        raise Exception("LOCALAPPDATA environment variable not found.")

    airsim_folder = os.path.join(local_appdata, 'AirSim')
    if not os.path.exists(airsim_folder):
        os.makedirs(airsim_folder)

    settings_path = os.path.join(airsim_folder, 'settings.json')

    settings = {
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "Vehicles": {
            "Drone1": {
                "VehicleType": "SimpleFlight",
                "AutoCreate": True
            }
        }
    }

    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=4)

    print("AirSim settings written to:", settings_path)


if __name__ == '__main__':
    write_settings()

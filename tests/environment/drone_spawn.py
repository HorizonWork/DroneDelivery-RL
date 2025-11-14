import airsim
import time

VEHICLE_NAME = "Drone1"

spawn_x = -60
spawn_y = 30
spawn_z = -10

yaw = 0
pitch = 0
roll = 0

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, VEHICLE_NAME)
client.armDisarm(True, VEHICLE_NAME)

pose = airsim.Pose(
    airsim.Vector3r(spawn_x, spawn_y, spawn_z), airsim.to_quaternion(pitch, roll, yaw)
)

client.simSetVehiclePose(pose, True, VEHICLE_NAME)

print(f"Drone spawned at: X={spawn_x} Y={spawn_y} Z={spawn_z}")

time.sleep(1)
client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()

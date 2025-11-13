import airsim
import time

# ---- CONFIG ----
# Tên vehicle trong settings.json (mặc định: "Drone1")
VEHICLE_NAME = "Drone1"

# Vị trí bạn muốn spawn drone (đơn vị: mét)
spawn_x = -60  # trục X UE
spawn_y = 30  # trục Y UE
spawn_z = -10
# trục Z (âm là bay trên không)

# Góc quay drone (đơn vị radian)
yaw = 0
pitch = 0
roll = 0

# ---- CONNECT ----
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, VEHICLE_NAME)
client.armDisarm(True, VEHICLE_NAME)

# ---- SET POSE ----
pose = airsim.Pose(
    airsim.Vector3r(spawn_x, spawn_y, spawn_z), airsim.to_quaternion(pitch, roll, yaw)
)

client.simSetVehiclePose(pose, True, VEHICLE_NAME)

print(f"Drone spawned at: X={spawn_x} Y={spawn_y} Z={spawn_z}")

# optional: takeoff
time.sleep(1)
client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()

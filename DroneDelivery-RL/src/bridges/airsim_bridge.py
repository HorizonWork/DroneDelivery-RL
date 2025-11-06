"""
AirSim connection interface
""",
import airsim

class AirSimBridge:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
    def connect(self):
        # Connect to AirSim
        pass
        
    def get_telemetry(self):
        # Get drone telemetry data
        pass
        
    def send_command(self, command):
        # Send command to drone
        pass
#!/usr/bin/env python3
"""
UE Position Helper Script
Get positions of objects in Unreal Engine via AirSim API
Use this to get accurate coordinates for DroneSpawn and Landing points
"""

import airsim
import sys
import json
from pathlib import Path

def get_object_position(client, object_name):
    """
    Get position of an object in UE.
    
    Args:
        client: AirSim client
        object_name: Name of object in UE (e.g., "DroneSpawn", "Landing_101")
    
    Returns:
        dict with x, y, z coordinates in AirSim NED frame
    """
    try:
        # Try to get object pose (if it's a spawned object)
        pose = client.simGetObjectPose(object_name)
        
        return {
            "name": object_name,
            "x": pose.position.x_val,
            "y": pose.position.y_val,
            "z": pose.position.z_val,
            "success": True
        }
    except Exception as e:
        return {
            "name": object_name,
            "error": str(e),
            "success": False
        }


def list_all_objects(client):
    """
    List all objects in the scene (if API supports it).
    """
    try:
        # Note: This method may not be available in all AirSim versions
        objects = client.simListSceneObjects()
        return objects
    except Exception as e:
        print(f"⚠️  Cannot list objects automatically: {e}")
        print("You'll need to manually enter object names from UE.")
        return []


def main():
    print("=" * 70)
    print("UE POSITION HELPER")
    print("=" * 70)
    print()
    
    # Connect to AirSim
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✅ Connected to AirSim")
        print()
    except Exception as e:
        print(f"❌ Failed to connect to AirSim: {e}")
        print("Make sure AirSim (UE) is running!")
        sys.exit(1)
    
    # Default objects to query
    default_objects = [
        "DroneSpawn",
        "Landing_101",
        "Landing_102",
        "Landing_103",
        "Landing_104",
        "Landing_105",
        "Landing_106"
    ]
    
    print("🔍 Attempting to list scene objects...")
    scene_objects = list_all_objects(client)
    if scene_objects:
        print(f"Found {len(scene_objects)} objects in scene:")
        for obj in scene_objects[:20]:  # Show first 20
            print(f"  - {obj}")
        print()
    
    print("📍 Querying default object positions...")
    print()
    
    positions = {}
    
    for obj_name in default_objects:
        result = get_object_position(client, obj_name)
        
        if result["success"]:
            print(f"✅ {obj_name}:")
            print(f"   X: {result['x']:.2f}")
            print(f"   Y: {result['y']:.2f}")
            print(f"   Z: {result['z']:.2f}")
            positions[obj_name] = result
        else:
            print(f"❌ {obj_name}: {result['error']}")
        print()
    
    # Get drone's current position as reference
    try:
        drone_state = client.getMultirotorState()
        drone_pos = drone_state.kinematics_estimated.position
        print("🚁 Current Drone Position (reference):")
        print(f"   X: {drone_pos.x_val:.2f}")
        print(f"   Y: {drone_pos.y_val:.2f}")
        print(f"   Z: {drone_pos.z_val:.2f}")
        print()
        
        positions["CurrentDrone"] = {
            "name": "CurrentDrone",
            "x": drone_pos.x_val,
            "y": drone_pos.y_val,
            "z": drone_pos.z_val
        }
    except Exception as e:
        print(f"⚠️  Could not get drone position: {e}")
        print()
    
    # Save to JSON file
    output_file = Path(__file__).parent.parent.parent / "config" / "ue_positions.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(positions, f, indent=2)
    
    print("=" * 70)
    print(f"✅ Positions saved to: {output_file}")
    print("=" * 70)
    print()
    print("📝 NEXT STEPS:")
    print("1. Check ue_positions.json for coordinates")
    print("2. Copy coordinates to config/landing_101_config.yaml")
    print("3. Remember: Add +1.2 to Z for landing hover point!")
    print()
    print("Example:")
    print("  positions:")
    print("    spawn_point:")
    print("      x: <DroneSpawn X>")
    print("      y: <DroneSpawn Y>")
    print("      z: <DroneSpawn Z>")
    print("    landing_points:")
    print("      - name: Landing_101")
    print("        x: <Landing_101 X>")
    print("        y: <Landing_101 Y>")
    print("        z: <Landing_101 Z + 1.2>  # Hover point above platform")
    print()


if __name__ == "__main__":
    main()

import os
import sys
import carla

# Read the working host
with open('/workspace/carla_host.txt', 'r') as f:
    host = f.read().strip()

print(f"Connecting to CARLA at {host}:2000...")
client = carla.Client(host, 2000)
client.set_timeout(10.0)

world = client.get_world()
print(f"Connected to world: {world.get_map().name}")

# Get a spawn point and spawn a vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_points = world.get_map().get_spawn_points()

if spawn_points:
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    print(f"Spawned vehicle: {vehicle.type_id}")
    
    # Clean up
    vehicle.destroy()
    print("Vehicle destroyed")
else:
    print("No spawn points available!")

print("Basic CARLA test passed!")

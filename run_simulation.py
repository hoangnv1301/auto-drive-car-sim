#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Run the autonomous driving simulation')
    parser.add_argument('--scenario', type=str, default='curved_road',
                        help='Scenario to load (default: curved_road)')
    parser.add_argument('--resolution', type=str, default='1280x720',
                        help='Display resolution (default: 1280x720)')
    parser.add_argument('--fullscreen', action='store_true',
                        help='Run in fullscreen mode')
    parser.add_argument('--autonomous', action='store_true',
                        help='Enable autonomous mode for the car')
    return parser.parse_args()

def create_default_scenario(scenario_name, scenario_file):
    """Create a default scenario file if it doesn't exist."""
    # Create a basic default scenario
    if scenario_name == 'lane_keeping':
        scenario_data = {
            'name': 'Lane Keeping Demo',
            'description': 'A simple lane keeping demo with a single car',
            'objects': [
                # Main car (will be autonomous)
                {
                    'type': 'car',
                    'position': [-80, 2, 0.75],  # Start slightly off center
                    'rotation': [0, 0, 0],
                    'velocity': [10, 0, 0],
                    'dimensions': [4.5, 2.0, 1.5],
                    'autonomous': True
                }
            ],
            'road_network': {
                'roads': [
                    {
                        'id': 0,
                        'start': [-100, 0, 0],
                        'end': [100, 0, 0],
                        'width': 10,
                        'lanes': 2
                    }
                ]
            }
        }
    elif scenario_name == 'curved_road':
        scenario_data = {
            'name': 'Curved Road Demo',
            'description': 'A scenario with a curved road to test vehicle\'s ability to detect and follow road curves',
            'objects': [
                # Main car (will be autonomous)
                {
                    'type': 'car',
                    'position': [-100, 2.0, 0.75],
                    'rotation': [0, 0, 0],
                    'velocity': [10, 0, 0],
                    'dimensions': [2.5, 1.7, 1.5],
                    'autonomous': True
                },
                # Stationary car as obstacle
                {
                    'type': 'car',
                    'position': [-40, -1.0, 0.75],
                    'rotation': [0, 0, 0],
                    'velocity': [0, 0, 0],
                    'dimensions': [4.5, 2.0, 1.5],
                    'autonomous': False
                },
                # Moving car
                {
                    'type': 'car',
                    'position': [-75, 2.5, 0.75],
                    'rotation': [0, 0, 0],
                    'velocity': [3.5, 0, 0],
                    'dimensions': [4.0, 1.8, 1.5],
                    'autonomous': False
                }
            ],
            'road_network': {
                'roads': [
                    {
                        'start': [-100, 0, 0],
                        'end': [100, 20, 0],  # End Y coordinate is 20 to create a curve
                        'width': 10,
                        'lanes': 2
                    }
                ]
            }
        }
    elif scenario_name == 'basic_intersection':
        scenario_data = {
            'name': 'Basic Intersection',
            'description': 'A simple four-way intersection with various vehicles',
            'objects': [
                # Cars
                {
                    'type': 'car',
                    'position': [0, -5, 0],
                    'rotation': [0, 0, 0],
                    'velocity': [0, 5, 0],
                    'dimensions': [4.5, 2.0, 1.5],
                    'autonomous': False
                },
                {
                    'type': 'car',
                    'position': [20, 3, 0],
                    'rotation': [0, 0, 3.14],  # Facing the opposite direction
                    'velocity': [-5, 0, 0],
                    'dimensions': [4.5, 2.0, 1.5],
                    'autonomous': False
                },
                # Trucks
                {
                    'type': 'truck',
                    'position': [-15, -3, 0],
                    'rotation': [0, 0, 0],
                    'velocity': [3, 0, 0],
                    'dimensions': [8.0, 2.5, 3.0]
                },
                {
                    'type': 'truck',
                    'position': [5, 15, 0],
                    'rotation': [0, 0, -1.57],  # Facing south
                    'velocity': [0, -4, 0],
                    'dimensions': [8.0, 2.5, 3.0]
                },
                # Tricar
                {
                    'type': 'tricar',
                    'position': [12, -8, 0],
                    'rotation': [0, 0, 0.78],  # ~45 degrees
                    'velocity': [2, 2, 0],
                    'dimensions': [3.0, 1.5, 1.5]
                },
                # Cyclists
                {
                    'type': 'cyclist',
                    'position': [-8, 7, 0],
                    'rotation': [0, 0, 2.35],  # ~135 degrees
                    'velocity': [-1, -1, 0],
                    'dimensions': [1.8, 0.8, 1.7]
                },
                # Pedestrians
                {
                    'type': 'pedestrian',
                    'position': [3, 3, 0],
                    'rotation': [0, 0, 3.9],  # ~225 degrees
                    'velocity': [-0.5, -0.5, 0],
                    'dimensions': [0.5, 0.5, 1.7]
                },
                {
                    'type': 'pedestrian',
                    'position': [-5, -2, 0],
                    'rotation': [0, 0, 1.57],  # 90 degrees
                    'velocity': [0, 0.5, 0],
                    'dimensions': [0.5, 0.5, 1.7]
                }
            ],
            'road_network': {
                'intersections': [
                    {
                        'id': 0,
                        'position': [0, 0, 0],
                        'connections': [0, 1, 2, 3]
                    }
                ],
                'roads': [
                    {
                        'id': 0,
                        'start': [-50, 0, 0],
                        'end': [50, 0, 0],
                        'width': 8,
                        'lanes': 2
                    },
                    {
                        'id': 1,
                        'start': [0, -50, 0],
                        'end': [0, 50, 0],
                        'width': 8,
                        'lanes': 2
                    }
                ]
            }
        }
    else:
        # Generic empty scenario
        scenario_data = {
            'name': scenario_name.replace('_', ' ').title(),
            'description': 'An empty scenario',
            'objects': [],
            'road_network': {
                'intersections': [],
                'roads': []
            }
        }
    
    # Save the scenario file
    with open(scenario_file, 'w') as f:
        json.dump(scenario_data, f, indent=2)
    
    print(f"Created default scenario: {scenario_name}")

def main():
    args = parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Make sure the assets directory exists
    assets_dir = os.path.join(script_dir, 'assets')
    scenarios_dir = os.path.join(assets_dir, 'scenarios')
    os.makedirs(scenarios_dir, exist_ok=True)
    
    # Check if the scenario file exists, create it if it doesn't
    scenario_file = os.path.join(scenarios_dir, f"{args.scenario}.json")
    if not os.path.exists(scenario_file):
        create_default_scenario(args.scenario, scenario_file)
    
    # For lane_keeping or curved_road scenarios, we always want the car to be autonomous
    if args.scenario in ['lane_keeping', 'curved_road']:
        args.autonomous = True
    
    # If autonomous mode is enabled and we're not using one of the auto-autonomous scenarios,
    # run the update_scenario script to set the first car as autonomous
    if args.autonomous and args.scenario not in ['lane_keeping', 'curved_road']:
        print("Enabling autonomous mode for the first car...")
        update_script = os.path.join(script_dir, 'src', 'update_scenario.py')
        subprocess.run([sys.executable, update_script, args.scenario])
    
    # Run the main simulation
    main_script = os.path.join(script_dir, 'src', 'main.py')
    cmd = [sys.executable, main_script,
           '--scenario', args.scenario,
           '--resolution', args.resolution]
    
    if args.fullscreen:
        cmd.append('--fullscreen')
    
    # Run the simulation
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    
    print("Simulation ended.")

if __name__ == "__main__":
    main() 
import os
import json

def load_scenario(scenario_name):
    """Load a scenario from file.
    
    Args:
        scenario_name (str): Name of the scenario to load
        
    Returns:
        dict: Scenario data
    """
    # Check if the scenario file exists
    scenario_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               'assets', 'scenarios')
    scenario_file = os.path.join(scenario_dir, f"{scenario_name}.json")
    
    # If the scenario file doesn't exist, create it with default data
    if not os.path.exists(scenario_file):
        os.makedirs(scenario_dir, exist_ok=True)
        scenario_data = _create_default_scenario(scenario_name)
        with open(scenario_file, 'w') as f:
            json.dump(scenario_data, f, indent=2)
        return scenario_data
    
    # Load the scenario from file
    with open(scenario_file, 'r') as f:
        return json.load(f)

def _create_default_scenario(scenario_name):
    """Create a default scenario if none exists.
    
    Args:
        scenario_name (str): Name of the scenario
        
    Returns:
        dict: Default scenario data
    """
    if scenario_name == 'curved_road_test':
        return {
            'name': 'Curved Road Test',
            'description': 'A single car on a road with a natural curve to test lane keeping',
            'objects': [
                # Main autonomous car - positioned at the start of the road
                {
                    'type': 'car',
                    'position': [-140, 0, 0],  # At the beginning of the straight segment
                    'rotation': [0, 0, 0],     # Facing forward along x-axis
                    'velocity': [3, 0, 0],     # Initial velocity along road
                    'dimensions': [4.5, 2.0, 1.5],
                    'autonomous': True         # Enable autonomous driving
                }
            ],
            'road_network': {
                'roads': [
                    {
                        'id': 0,
                        'width': 10.0,
                        'lanes': 1
                    }
                ]
            }
        }
    elif scenario_name == 'basic_intersection':
        return {
            'name': 'Basic Intersection',
            'description': 'A simple four-way intersection with various vehicles',
            'objects': [
                # Cars
                {
                    'type': 'car',
                    'position': [0, -5, 0],
                    'rotation': [0, 0, 0],
                    'velocity': [0, 5, 0],
                    'dimensions': [4.5, 2.0, 1.5]
                },
                {
                    'type': 'car',
                    'position': [20, 3, 0],
                    'rotation': [0, 0, 3.14],  # Facing the opposite direction
                    'velocity': [-5, 0, 0],
                    'dimensions': [4.5, 2.0, 1.5]
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
        return {
            'name': scenario_name.replace('_', ' ').title(),
            'description': 'An empty scenario',
            'objects': [],
            'road_network': {
                'intersections': [],
                'roads': []
            }
        } 
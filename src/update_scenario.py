#!/usr/bin/env python3
import os
import json
import sys

def update_scenario(scenario_name='basic_intersection'):
    """Update a scenario to set some vehicles as autonomous.
    
    Args:
        scenario_name (str): Name of the scenario to update
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Get the scenario file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    scenario_dir = os.path.join(root_dir, 'assets', 'scenarios')
    scenario_file = os.path.join(scenario_dir, f"{scenario_name}.json")
    
    # Check if the file exists
    if not os.path.exists(scenario_file):
        print(f"Error: Scenario file '{scenario_file}' not found.")
        return False
    
    # Load the scenario
    try:
        with open(scenario_file, 'r') as f:
            scenario_data = json.load(f)
    except Exception as e:
        print(f"Error loading scenario file: {e}")
        return False
    
    # Update cars to be autonomous
    updated = False
    for obj in scenario_data.get('objects', []):
        if obj.get('type') == 'car':
            obj['autonomous'] = True
            updated = True
            print(f"Set car at position {obj['position']} to autonomous mode.")
            # Just update the first car and break
            break
    
    if not updated:
        print("No cars found in the scenario to update.")
        return False
    
    # Save the updated scenario
    try:
        os.makedirs(scenario_dir, exist_ok=True)
        with open(scenario_file, 'w') as f:
            json.dump(scenario_data, f, indent=2)
        print(f"Successfully updated scenario '{scenario_name}'.")
        return True
    except Exception as e:
        print(f"Error saving scenario file: {e}")
        return False

if __name__ == "__main__":
    # Get scenario name from command line argument, if provided
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else 'basic_intersection'
    success = update_scenario(scenario_name)
    sys.exit(0 if success else 1) 